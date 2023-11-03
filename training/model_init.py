from models import MLP, RNN, LSTM
from training import word_embedding
from embedding import vectorise


def model_fn(inputs: dict, params: dict):
    model_type = params.get("model", "").upper()
    max_features = params.get("max_features", 10**4)

    if "BERT" in model_type:
        # Sentence BERT embeddings
        if model_type.endswith("MLP"):
            # TODO: MLP SBERT Embeddings
            # TODO: MLP SBERT Model
            pass
        elif model_type.endswith("RNN"):
            # TODO: RNN SBERT Embeddings
            # TODO: RNN SBERT Model
            pass
        elif model_type.endswith("LSTM"):
            # TODO: LSTM SBERT Embeddings
            # TODO: LSTM SBERT Model
            pass
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
            results = word_embedding.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=MLP.MLP,
                # params=test_params,
                params=params,
            )
        elif model_type.endswith("RNN"):
            results = word_embedding.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=RNN.RNN,
                # params=test_params,
                params=params,
            )
        elif model_type.endswith("LSTM"):
            results = word_embedding.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=LSTM.BaselineLSTM,
                # params=test_params,
                params=params,
            )
        else:
            raise NotImplementedError

    return results
