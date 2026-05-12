class BaseModel:
    def predict(self, waveform, sample_rate, prompt):
        raise NotImplementedError("Subclasses must implement predict().")