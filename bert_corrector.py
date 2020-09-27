# -*- coding: utf-8 -*-
import operator
import sys
import time

from transformers import pipeline

sys.path.append('../..')
import config
from text_utils import is_chinese_string, convert_to_unicode
from logger import logger
from corrector import Corrector


class BertCorrector(Corrector):
    def __init__(self, bert_model_dir=config.bert_model_dir,
                 bert_config_path=config.bert_config_path,
                 bert_model_path=config.bert_model_path):
        super(BertCorrector, self).__init__()
        self.name = 'bert_corrector'
        t1 = time.time()
        self.model = pipeline('fill-mask',
                              model=bert_model_path,
                              config=bert_config_path,
                              tokenizer=bert_model_dir)
        if self.model:
            self.mask = self.model.tokenizer.mask_token
            logger.debug('Loaded bert model: %s, spend: %.3f s.' % (bert_model_dir, time.time() - t1))

    def bert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = self.split_2_short_text(text, include_symbol=True)  # 获得短句的及其起始位置
        for blk, start_idx in blocks:
            blk_new = ''
            for idx, s in enumerate(blk):
                # 对非中文的错误不做处理
                if is_chinese_string(s):
                    sentence_lst = list(blk_new + blk[idx:])
                    sentence_lst[idx] = self.mask
                    sentence_new = ''.join(sentence_lst)
                    predicts = self.model(sentence_new)
                    top_tokens = []
                    for p in predicts:
                        token_id = p.get('token', 0)
                        token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                        top_tokens.append(token_str)  # 得到可能替换的词

                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)  # 获得候选词
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
            # print("text_new",text_new)
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    d = BertCorrector()
    error_sentencess = {
        "奇石店": "起始点",
    }

    for sent in error_sentencess.keys():


        restic1 = time.clock()
        corrected_sent, err = d.bert_correct(sent)
        print("耗时 %s seconds" % (time.clock()-restic1))
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))
    import datetime
    for sent in error_sentencess.keys():

        tic = datetime.datetime.now()
        corrected_sent, err = d.bert_correct(sent)
        end = datetime.datetime.now()
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))
        print("耗时 %s seconds" % (end - tic))
