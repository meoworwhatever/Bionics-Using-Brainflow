import asyncio
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from fastapi import FastAPI
from socketio import AsyncServer, ASGIApp
import uvicorn

from kinematics_engine import LegKinematicsEngine
from safety_guard import SafetyGuard
from hardware_interface import BionicHardwareInterface

BOARD_ID = BoardIds.SYNTHETIC_BOARD.value 
PARAMS = BrainFlowInputParams()
BOARD = BoardShim(BOARD_ID, PARAMS)

# (0.1 = very smooth but slow, 0.9 = twitchy but fast)
SMOOTHING_ALPHA = 0.15 
# if the signal is too loud, ignore it.
ARTIFACT_THRESHOLD = 150.0 

baseline_rms = 5.0
peak_rms = 30.0   
last_smoothed_flex = 0.0

# server setup
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = ASGIApp(sio, app)

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
        # get only the last 250ms of data for low-latency response
        # 128 samples is roughly 0.5s at 250Hz—perfect for a rolling window
        data = BOARD.get_current_board_data(128)

        if data.any() and len(data[0]) > 10:
            eeg_signal = np.copy(data[target_channel])

            DataFilter.detrend(eeg_signal, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(eeg_signal, sampling_rate, 15.0, 30.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

            current_rms = np.std(eeg_signal)

            # don't move limb if user blinks or bites down
            if current_rms > ARTIFACT_THRESHOLD:
                raw_intent = last_smoothed_flex 
            else:
                raw_intent = calculate_flex(current_rms)
                
            last_smoothed_flex = (SMOOTHING_ALPHA * raw_intent) + ((1 - SMOOTHING_ALPHA) * last_smoothed_flex)

            await sio.emit('neural_intent', {'flex': round(last_smoothed_flex, 3)})

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
