import asyncio
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from fastapi import FastAPI
from socketio import AsyncServer, ASGIApp
import uvicorn

from kinematics_engine import LegKinematicsEngine, ArmKinematicsEngine
from safety_guard import SafetyGuard
from hardware_interface import BionicHardwareInterface

BOARD_ID = BoardIds.SYNTHETIC_BOARD.value 
PARAMS = BrainFlowInputParams()
BOARD = BoardShim(BOARD_ID, PARAMS)

# (0.1 = very smooth but slow, 0.9 = twitchy but fast)
SMOOTHING_ALPHA = 0.15 
# If the signal is too loud, ignore it.
ARTIFACT_THRESHOLD = 150.0 

baseline_rms = 5.0
peak_rms = 30.0   
last_smoothed_flex = 0.0

# Server setup
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = ASGIApp(sio, app)

leg_kinematics = LegKinematicsEngine()
arm_kinematics = ArmKinematicsEngine()
guard = SafetyGuard()

hw_interface = BionicHardwareInterface()
hw_interface.connect()

def calculate_flex(rms):
    """Maps raw brain energy (RMS) to a 0.0 - 1.0 range with a deadzone."""
    global baseline_rms, peak_rms
    
    range_val = peak_rms - baseline_rms
    if range_val <= 0: range_val = 1 
    
    raw_flex = (rms - baseline_rms) / range_val
    
    # ignore tiny micro-movements to keep the motors quiet
    if raw_flex < 0.15:
       return 0.0
    
    return min(max(raw_flex, 0.0), 1.0)

async def stream_neural_data():
    global last_smoothed_flex
    
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    target_channel = eeg_channels[0] 

    while True:
        # Get only the last 250ms of data for low-latency response
        data = BOARD.get_current_board_data(128)

        if data.any() and len(data[0]) > 10:
            eeg_signal = np.copy(data[target_channel])

            DataFilter.detrend(eeg_signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(eeg_signal, sampling_rate, 15.0, 30.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

            current_rms = np.std(eeg_signal)

            # Don't move limb if user blinks or bites down
            if current_rms > ARTIFACT_THRESHOLD:
                raw_intent = last_smoothed_flex 
            else:
                raw_intent = calculate_flex(current_rms)
                
            last_smoothed_flex = (SMOOTHING_ALPHA * raw_intent) + ((1 - SMOOTHING_ALPHA) * last_smoothed_flex)
            
            # LEG TRACKING CONTEXT 
            raw_leg_angles = leg_kinematics.process_intent_to_angles(last_smoothed_flex)
            is_leg_safe, safe_knee = guard.validate_movement(raw_leg_angles["knee"], current_rms)
            leg_ratio = safe_knee / max(0.001, raw_leg_angles["knee"])
            
            sanitized_leg = {
                "hip_pitch": raw_leg_angles["hip_pitch"] * leg_ratio,
                "knee": safe_knee,
                "ankle_pitch": raw_leg_angles["ankle_pitch"] * leg_ratio
            }

            # ARM TRACKING CONTEXT 
            raw_arm_angles = arm_kinematics.process_intent_to_angles(last_smoothed_flex)
            is_arm_safe, safe_elbow = guard.validate_movement(raw_arm_angles["elbow"], current_rms)
            arm_ratio = safe_elbow / max(0.001, raw_arm_angles["elbow"])
            
            sanitized_arm = {
                "shoulderX": raw_arm_angles["shoulderX"] * arm_ratio,
                "elbow": safe_elbow,
                "wristX": raw_arm_angles["wristX"] * arm_ratio
            }

            # HARDWARE DISPATCH & TWIN BROADCAST ---
            # merging both limb angle dictionaries together for processing
            sanitized_full_body = {**sanitized_leg, **sanitized_arm}
            twin_data = hw_interface.write_actuators(sanitized_full_body)
            
            # more on the global telemetry parameters 
            twin_data["flex"] = round(last_smoothed_flex, 3)
            twin_data["rms"] = round(current_rms, 2)
            
            await sio.emit('neural_intent', twin_data)

        await asyncio.sleep(0.04) # 25 FPS update rate

@sio.on('connect')
async def connect(sid, environ):
    print(f"Bionic Dashboard connected: {sid}")

if __name__ == "__main__":
    BOARD.prepare_session()
    BOARD.start_stream()
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(stream_neural_data())
        uvicorn.run(socket_app, host="0.0.0.0", port=5000)
    finally:
        BOARD.stop_stream()
        BOARD.release_session()
