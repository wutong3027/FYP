from transformers import AutoModelForSeq2SeqLM, BartTokenizer
from ArticleSummarization.classes.machineLearning import MachineLearning

class NeuralNetwork(MachineLearning):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    def NN_generate_summary(self, text):
        tokenizer = self.tokenizer
        model = self.model
        if text is not None:
            input_ids = tokenizer(text, truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
            summary_ids = model.generate(input_ids['input_ids'], num_beams=10, max_length=512)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary