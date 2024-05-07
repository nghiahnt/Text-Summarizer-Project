from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline():
    def __init__(self, levelValue = 128):
        self.levelValue = levelValue
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text, levelValue):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": levelValue}

        pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output