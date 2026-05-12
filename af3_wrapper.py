import torch
import numpy as np
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Processor

class AudioFlamingoInterface:
    def __init__(self, model_id="nvidia/audio-flamingo-3-hf", device="auto"):
        self.processor = AudioFlamingo3Processor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16
        )

        self.model.eval()
        self.device = self.model.device

    def query(self, audio_array, prompt, sampling_rate=16000, **gen_kwargs):
        """
        Sends a single audio array and a text prompt to the model.
        """
        assert sampling_rate == 16000
        # Ensure audio is a numpy array for the processor
        if torch.is_tensor(audio_array):
            audio_np = audio_array.cpu().numpy()
        else:
            audio_np = audio_array

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "audio": audio_np},
                    ],
                }
            ]
        ]

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)

        # Process inputs
        # inputs = self.processor(text=prompt, audio=audio_np, return_tensors='pt').to(self.device)
        # --- FIX: Cast input features to the model's precision (float16) ---
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode only the newly generated tokens
        input_token_len = inputs.input_ids.shape[1]
        decoded = self.processor.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True
        )[0].strip()

        return decoded


from models.base import BaseModel

class AudioFlamingoModel(BaseModel):

    def __init__(self, temperature=0.2, max_new_tokens=40, **kwargs):

        self.model = AudioFlamingoInterface()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def predict(self, waveform, sample_rate, prompt):

        return self.model.query(
            audio_array=waveform,
            prompt=prompt,
            sampling_rate=sample_rate,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0
        )
