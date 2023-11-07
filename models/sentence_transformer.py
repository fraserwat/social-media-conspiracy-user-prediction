import torch
from transformers import BertTokenizer, BertModel


class SentenceTransformer(torch.nn.Module):
    def __init__(
        self, model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", **kwargs
    ):
        super().__init__()
        # loads the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, **kwargs)
        # loads the BertModel
        self.model = BertModel.from_pretrained(model_name_or_path, **kwargs)

    def forward(self, inputs):
        # runs tokenization and creates embedding
        tokenized = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )

        # Passes the input tensors to the BERT model
        outputs = self.model(
            input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
        )

        # 'input_ids' and 'attention_mask' are used in the forward pass of a BERT model
        # Applies mean pooling to the last hidden states to get sentence embeddings
        input_mask_expanded = (
            tokenized["attention_mask"]
            .unsqueeze(-1)
            .expand(outputs.last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)

        # results in a tensor that holds the count of actual (non-padded) tokens in each sequence.
        sum_mask = input_mask_expanded.sum(1)

        # Apply clamping to sum_mask to avoid divide by zero when calculating mean embeddings.
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Calculates mean only over actual tokens. Ignore padded tokens as embeddings zeroed out.
        # result is tensor of mean embeddings for each sequence in batch.

        # Commonly used in NLP tasks to generate fixed-len sentence embeddings from transformers
        # outputs. Allows model to handle diff sequence lens + calculate sentence representations.
        return sum_embeddings / sum_mask
