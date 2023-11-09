import argparse
from models import MLP, RNN, LSTMBaseline, LSTMSentence
from training import word_models, sentence_models
from embedding import vectorise


def model_fn(inputs: dict, params: argparse.Namespace):
    model_type = params.model.upper()
    max_features = params.max_features

    if "BERT" in model_type:
        bert_model = params.transformer_model
        if model_type.endswith("MLP"):
            results = sentence_models.sentence_embedded_model(
                input_data=inputs,
                transformer_model=bert_model,
                model=MLP.MLP,
                params=params,
            )
        elif model_type.endswith("RNN"):
            results = sentence_models.sentence_embedded_model(
                input_data=inputs,
                transformer_model=bert_model,
                model=RNN.RNN,
                params=params,
            )
        elif model_type.endswith("LSTM"):
            results = sentence_models.sentence_embedded_model(
                input_data=inputs,
                transformer_model=bert_model,
                model=LSTMSentence.SentenceLSTM,
                params=params,
            )
        else:
            raise NotImplementedError
    else:
        # Word embeddings - as we are embedding manually, need to create vectorised layer
        vectorise_fn = vectorise.create_vectorised_layer(
            # inputs["train"][0] will be the first item in the dict, words_train
            words=inputs["train"][0],
            max_features=max_features,
        )

        if model_type.endswith("MLP"):
            results = word_models.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=MLP.MLP,
                params=params,
            )
        elif model_type.endswith("RNN"):
            results = word_models.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=RNN.RNN,
                params=params,
            )
        elif model_type.endswith("LSTM"):
            results = word_models.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=LSTMBaseline.BaselineLSTM,
                params=params,
            )
        else:
            raise NotImplementedError

    return results
