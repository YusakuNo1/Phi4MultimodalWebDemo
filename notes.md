# Model
https://huggingface.co/microsoft/Phi-4-multimodal-instruct

# External Environment & Model Hacks Summary

**1. Hugging Face Cache Layer (Core Model Patch)**
* **Target File:** `~/.cache/huggingface/modules/transformers_modules/.../modeling_phi4mm.py`
* **The Problem:** The `peft` library fails to initialize the vision adapter because the inner `Phi4MMModel` class lacks the `prepare_inputs_for_generation` method required by the wrapper.
* **The Hack:** Manually locate the line `peft_model = get_peft_model(self.model, vision_lora_config, adapter_name="vision")` and insert the following line immediately **before** it:
  `self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation`
* **Result:** Fixes the `AttributeError` during model initialization.

**2. Conda Environment Layer (Dependency Resolution)**
* **Target Tool:** `PyAudio` installation.
* **The Problem:** Standard `pip install` fails due to GCC/G++ compiler conflicts with system headers (missing `bits/timesize.h`) within the Miniconda toolchain.
* **The Hack:** Use the pre-compiled binary provided by the Conda-Forge channel:
  `conda install -c conda-forge pyaudio`
* **Result:** Bypasses the local C compilation phase and resolves library link errors in WSL.

**3. OS / WSL Layer (Audio Routing & Decoding)**
* **Target Tools:** `FFmpeg` and Windows Privacy Settings.
* **The Problem:** WSL cannot natively decode browser-based audio containers (WebM/Opus), and the virtualized Linux environment requires explicit permission to access Windows hardware.
* **The Hack:** * Install system-level decoders: `sudo apt install ffmpeg`.
    * Enable "Let desktop apps access your microphone" in Windows Privacy & Security settings.
* **Result:** Allows `pydub` to utilize `ffmpeg` for robust audio decoding and enables audio passthrough via WSLg.

**4. Transformers Library Layer (API Compatibility)**
* **Target Class:** `transformers.cache_utils.DynamicCache`
* **The Problem:** Phi-4's remote code calls `get_usable_length()`, a method that was renamed or modified in recent versions of the `transformers` library (v4.48+).
* **The Hack:** Perform a "Monkey Patch" on the global class at the start of execution:
  `DynamicCache.get_usable_length = lambda self, seq_len, layer_idx=None: self.get_seq_length(layer_idx)`
* **Result:** Restores backward compatibility for the model's KV-Cache management.