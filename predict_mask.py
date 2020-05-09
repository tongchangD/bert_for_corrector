# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from transformers import pipeline

MASK_TOKEN = "[MASK]"


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model_dir", default='./data/bert_models/chinese_finetuned_lm/',
                        type=str,
                        help="Bert pre-trained model dir")
    parser.add_argument("--bert_model_path", default='./data/bert_models/chinese_finetuned_lm/pytorch_model.bin',
                        type=str,
                        help="Bert pre-trained model path")
    parser.add_argument("--bert_config_path", default='./data/bert_models/chinese_finetuned_lm/bert_config.json',
                        type=str,
                        help="Bert pre-trained model config path")
    args = parser.parse_args()
    print("args",args)
    nlp = pipeline('fill-mask',
                   model=args.bert_model_path,
                   config=args.bert_config_path,
                   tokenizer=args.bert_model_dir
                   )
    print("nlp",nlp)
    for i in nlp('[MASK]买的电脑要缴税吗'):
        print(i)


if __name__ == "__main__":
    main()
