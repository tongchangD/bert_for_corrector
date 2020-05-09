# -*- coding: utf-8 -*-
import operator
import sys
import time

from transformers import pipeline

sys.path.append('../..')
import config
from text_utils import is_chinese_string, convert_to_unicode
from logger import logger
from pycorrector.corrector import Corrector


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
        "宁静多少东西可以免税": "入境多少东西可以免税",
        "自主通道在哪": "自助通道在哪",
        "后天半夜十二点过去原味执行人脸识别": "后天半夜十二点过去原位执行人脸识别",
        "智能判读": "智能判图",
        "上午十点45停止低温检测": "上午十点45停止低温检测",
        "星期四临晨两点前往判读是做低温探测": "星期四凌晨两点前往判图室做低温探测",
        "怎么确定形容渠道静静的金属制品的完税价格": "怎么确定行邮渠道进境的金属制品的完税价格",
        "看右手": "抬左手",
        "给经理经理": "给经理敬礼",
        "今天中午两点执行人连对比": "今天中午两点执行人脸比对",
        "培养的细菌可以促进吗": "培养的细菌可以出境吗",
        "申报太原在哪里": "申报台在哪里",
        "去什么胎该怎么走": "去申报台该怎么走",
        "跟谁": "跟随",
        "借助采集": "自助采集",
        "如何有效预防防热病": "如何有效预防黄热病",
        "货源": "复原",
        "带我去舔卡牌": "带我去填卡台",
        "今天凌晨一点带我去制度充电": "今天凌晨一点带我去自助通道",
        "星期五凌晨两点开始去雨卜空": "星期五凌晨两点开始区域布控",
        "通道附近能改变一张吗": "通道附近能盖验讫章吗",
        "绝代烟丝超过多少克需要填写申报单": "携带烟丝超过多少克需要填写申报单",
        "可以写的香蕉入境吗": "可以携带香蕉入境吗",
        "可以游记人民币出境吗": "可以邮寄人民币出境吗",
        "请问可不可以带吃的经济呀": "请问可不可以带吃的进境呀",
        "陕西多次往返的意思是什么": "短期多次往返的意思是什么",
        "可以带信子入境吗": "可以带杏子入境吗",
        "星期天半夜一点但我去查砚台": "星期天半夜一点带我去查验台",
        "遇见多少东西可以免税": "入境多少东西可以免税",
        "自助通道入境能干一七章吗": "自助通道入境能盖验讫章吗",
        "舔打开": "填卡台",
        "关闭玉不空": "关闭区域布控",
        "打开区域不可": "打开区域布控",
        "烧麦": "小麦",
        "阜丰路的税率是多少": "护肤露的税率是多少",
        "自助采摘": "自助采集",
        "星期三下午四点到监控点执行人员比对": "星期三下午四点到监控点执行人脸比对",
        "天罚出入境通行证的审批依据是什么": "填报出入境通行证的审批依据是什么",
        "膝盖东西超过免税金额如何处理": "携带东西超过免税金额如何处理",
        "显卡台在哪个方向": "填卡台在哪个方向",
        "鞋带英诗超过多少克需要填写申报单": "携带烟丝超过多少克需要填写申报单",
        "励志停止低温探测": "立即停止低温探测",
        "关闭区域不好": "关闭区域布控",
        "洗手机": "洗手间",
        "我想问问你制度通道在哪里": "我想问问你自助通道在哪里",
        "金牛雾查询": "截留物查询",
        "打开定位探测": "打开低温探测",
        "下午3点26分去世报台": "下午3点26分去申报台",
        "今天半夜一点带我去原味": "今天半夜一点带我去原位",
        "可以携带冰糖露军吗": "可以携带冰糖入境吗",
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
