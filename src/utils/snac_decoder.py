# modules/snac_decoder.py

import time
import torch
import numpy as np
from src.utils.logging import logger
from snac import SNAC  # Ensure that the snac module is installed

# Load SNAC model
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using SNAC on device: %s", snac_device)
snac_model = snac_model.to(snac_device)
cuda_stream = torch.cuda.Stream() if snac_device == "cuda" else None

def convert_to_audio(multiframe, count):
    if len(multiframe) < 7:
        return None
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    for j in range(num_frames):
        idx = j * 7
        codes_0[j] = frame_tensor[idx]
        codes_1[j * 2] = frame_tensor[idx + 1]
        codes_1[j * 2 + 1] = frame_tensor[idx + 4]
        codes_2[j * 4] = frame_tensor[idx + 2]
        codes_2[j * 4 + 1] = frame_tensor[idx + 3]
        codes_2[j * 4 + 2] = frame_tensor[idx + 5]
        codes_2[j * 4 + 3] = frame_tensor[idx + 6]
    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    with stream_ctx, torch.inference_mode():
        audio_hat = snac_model.decode(codes)
        audio_slice = audio_hat[:, :, 2048:4096]
        if snac_device == "cuda":
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            audio_np = audio_slice.detach().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    if "<custom_token_" not in token_string:
        return None
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1 or not token_string.endswith(">"):
        return None
    try:
        number_str = token_string[last_token_start + 14:-1]
        return int(number_str) - 10 - ((index % 7) * 4096)
    except (ValueError, IndexError):
        return None

token_cache = {}
MAX_CACHE_SIZE = 1000

def tokens_decoder(token_gen):
    buffer = []
    count = 0
    min_frames_required = 28
    process_every = 7
    for token_text in token_gen:
        cache_key = (token_text, count % 7)
        if cache_key in token_cache:
            token = token_cache[cache_key]
        else:
            token = turn_token_into_id(token_text, count)
            if token is not None and len(token_cache) < MAX_CACHE_SIZE:
                token_cache[cache_key] = token
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if count % process_every == 0 and count >= min_frames_required:
                buffer_to_proc = buffer[-min_frames_required:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen):
    audio_segments = list(tokens_decoder(syn_token_gen))
    return b"".join(audio_segments)
