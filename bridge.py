import asyncio
import numpy as np
import uvicorn
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from fastapi import FastAPI
from socketio import AsyncServer, ASGIApp

baseline_rms = 5.0
peak_rms = 30.0
THROTTLING_FPS = 0.04

sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = ASGIApp(sio, app)

params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value 
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

def calculate_flex(rms):
    """Maps raw brain energy (RMS) to a 0.0 - 1.0 range based on calibration."""
    global baseline_rms, peak_rms
    
    range_val = peak_rms - baseline_rms
    if range_val <= 0: range_val = 1
    
    raw_flex = (rms - baseline_rms) / range_val
    
    if raw_flex < 0.1:
        return 0.0
    
    return min(max(raw_flex, 0.0), 1.0)

@sio.on('connect')
async def connect(sid, environ):
    print(f"Bionic Dashboard Connected: {sid}")

@sio.on('calibrate')
async def calibrate(sid, data):
    """
    Sets baseline or peak from the dashboard.
    Payload: {'type': 'baseline'} or {'type': 'peak'}
    """
    global baseline_rms, peak_rms
    
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    data_buffer = board.get_current_board_data(sampling_rate)
    current_rms = np.sqrt(np.mean(data_buffer[1]**2))
    
    if data.get('type') == 'baseline':
        baseline_rms = current_rms
        print(f"CALIBRATED: New Baseline is {baseline_rms:.2f}")
    elif data.get('type') == 'peak':
        peak_rms = current_rms
        print(f"CALIBRATED: New Peak is {peak_rms:.2f}")

async def stream_neural_data():
    while True:
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        data = board.get_current_board_data(256) 

        if data.any() and len(data[1]) > 0:
            eeg_channel = data[1] # Using Synthetic Channel 1

            DataFilter.detrend(eeg_channel, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                eeg_channel, sampling_rate, 19.0, 22.0, 4,
                FilterTypes.BUTTERWORTH.value, 0
            )

            rms_value = np.sqrt(np.mean(eeg_channel**2))

            intent_flex = calculate_flex(rms_value)

            await sio.emit('neural_intent', {'flex': intent_flex})

        await asyncio.sleep(THROTTLING_FPS)

if __name__ == "__main__":
    import uvicorn
    loop = asyncio.get_event_loop()
    loop.create_task(stream_neural_data())
    uvicorn.run(socket_app, host="0.0.0.0", port=5000)
