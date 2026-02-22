import torch
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForCausalLM, AutoProcessor
from fastapi.responses import HTMLResponse
from pydub import AudioSegment

app = FastAPI()

# --- Model initialization ---
model_id = "microsoft/phi-4-multimodal-instruct"
device = "cuda"

print("Loading model...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map=device, 
    torch_dtype="auto", 
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)


@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    # 1. More robust audio processing
    audio_stream = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(audio_stream).set_frame_rate(16000).set_channels(1)
    
    # Force normalization
    if audio.max > 0:
        audio = audio.normalize()
        
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # audio_data = samples / (2**15)

    # Key fix: normalize according to actual audio bit depth
    # pydub sample_width: 2 means 16-bit (max 32768)
    denominator = 2**(8 * audio.sample_width - 1)
    audio_data = samples / denominator

    # Debug print: peak amplitude should be between 0.1 and 0.9 for normal speech
    print(f"Peak amplitude after correction: {np.abs(audio_data).max():.4f}")

    # Print audio status, ensure it's not silent
    print(f"Audio peak amplitude: {np.abs(audio_data).max():.4f}")

    # 2. Construct prompt (use the tag that worked for you)
    # prompt = "<|user|><|audio_1|>Translate this audio to Chinese.<|assistant|>"
    prompt = "<|user|>Answer this question in Chinese<|audio_1|><|assistant|>"

    # 3. Prepare input
    inputs = processor(
        text=prompt, 
        audios=[(audio_data, 16000)], 
        return_tensors="pt"
    ).to("cuda")

    # Debug print: confirm input
    print(f"Input IDs Shape: {inputs['input_ids'].shape}")
    if 'input_audio_embeds' in inputs:
        print("Status: audio features successfully embedded in Tensor")

    # 4. Generation parameters optimized for translation
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,        # Enable sampling
            temperature=0.7,      # Add some creativity to prevent loops
            top_p=0.9,
            repetition_penalty=1.1 # Avoid repetition
        )
    
    # 5. Extract response
    input_len = inputs['input_ids'].shape[1]
    response = processor.batch_decode(
        generate_ids[:, input_len:], 
        skip_special_tokens=True
    )[0]

    return {"text": response.strip()}


@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
