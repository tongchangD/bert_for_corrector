# BERT model correct error character with mask feature

## 使用说明

1. 保存预训练模型在data文件夹下
├── data  
│   ├── bert_config.json  
│   ├── config.json  
│   ├── pytorch_model.bin  
│   └── vocab.txt  
├── bert_corrector.py  
├── config.py  
├── logger.py  
├── predict_mask.py  
├── README.md  
└── text_utils.py  

2. 运行`bert_corrector.py`可以进行纠错。
```
python3 bert_corrector.py
```
3. 评估
通用数据下训练的结果并不适用于垂直领域的纠错，需要重新训练
export CUDA_VISIBLE_DEVICES=0
python run_lm_finetuning.py \
    --output_dir=chinese_finetuned_lm \
    --model_type=bert \
    --model_name_or_path=bert-base-chinese \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm
    --num_train_epochs=3


