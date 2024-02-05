import argparse
import os.path

from components.gector.utils.helpers import read_lines
from components.gector.gector.gec_model import GecBERTModel

def load_for_demo(use_roberta=True, gpu_id=0):
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    if use_roberta:
        model_path = os.path.join(model_path, 'roberta_1_gector.th')
        transformer_model = 'roberta'
        special_tokens_fix = 1
        min_error_prob = 0.50
        confidence_bias = 0.20
    else:
        model_path = os.path.join(model_path, 'xlnet_0_gector.th')
        transformer_model = 'xlnet'
        special_tokens_fix = 0
        min_error_prob = 0.66
        confidence_bias = 0.35
    vocab_path = os.path.join(os.path.dirname(
        __file__), 'data', 'output_vocabulary', '')
    model = GecBERTModel(vocab_path=vocab_path,
                         model_paths=[model_path],
                         iterations=5,
                         model_name=transformer_model,
                         special_tokens_fix=special_tokens_fix,
                         min_error_probability=min_error_prob,
                         confidence=confidence_bias,
                         is_ensemble=0,
                         gpu_id=gpu_id)
    return model


# inference for demo
def predict_for_demo(lines, model, batch_size=32):
    test_data = [s.strip() for s in lines]
    predictions = []
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)

    # output = '<eos>'.join([' '.join(x) for x in predictions])
    output = [' '.join(x) for x in predictions]
    return output
