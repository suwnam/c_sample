import sys
import math

import numpy as np
import torch

from korbertTokenizer.tokenization import KorBertTokenizer
from models.bert_seq_cls import BERTSeqClsUqModel


class FaqClsPredictor:
    """ FAQ/OOD/Non-Acceptable Text 분류 모델.
        초기 버전은 KorBERT 음절 모델을 사용하였다.
    """
    def __init__(self, model_path, logger, ood_threshold=0.165):
        temp = 3/math.pow(math.pi, 2)

        self.logger = logger

        self.tokenizer = KorBertTokenizer.from_pretrained(model_path)
        self.model = BERTSeqClsUqModel(model_path + '/bert_config.json', '', 9)
        self.model.load_states(model_path + '/ep7_val0.037_model_states.pt')
        self.model.set_temperature(temp)

        if torch.cuda.is_available():
            self.logger.info('CUDA detected, assign model into cuda:0 device for faster inference.')
            self.compute_device = torch.device('cuda:0')
        else:
            self.logger.info('CUDA not available. assign model into CPU.')
            self.compute_device = torch.device('cpu')

        self.model = self.model.to(device=self.compute_device)
        self.model.eval()

        self.ood_threshold = ood_threshold

        self.logger.info("FAQ/OOD Classifier Model initialized. temperature T=%.4f, Threshold H=%.4f",
                temp, self.ood_threshold)

        # 9개의 classes
        self.TAG_NAMES = ['방문안내', '시설안내', '연구원안내', '회사소개', '외부API',
                          '기술소개', '로봇프로필', 'INSANE', 'OOD']

    def encode_texts(self, texts_or_text_pairs: list[str]):
        features = self.tokenizer.batch_encode_plus(
                texts_or_text_pairs, max_length=128, padding='max_length',
                truncation=True, return_tensors='pt')
        return features

    def inference(self, a_list_of_texts: list[str]):
        """ 반환되는 튜플은 (입력텍스트, 태그, 확률값, OOD 여부) 로 구성됨
        """
        ret_list = []
        input_feats = self.encode_texts(a_list_of_texts)
        with torch.no_grad():
            input_feats = input_feats.to(device=self.compute_device)
            outputs, gp_cov = self.model.forward(input_feats['input_ids'],
                    input_feats['token_type_ids'], input_feats['attention_mask'],
                    is_training=False, ret_gp_coverage=True, do_not_use_meanfield=False)

        probs = torch.softmax(outputs, axis=1)
        cl_prob, maxcl = torch.max(probs, axis=1)
        cl_prob = cl_prob.detach().cpu().numpy()
        maxcl = maxcl.detach().cpu().numpy()
        ood_prob = 1. - cl_prob

        for idx, a_text in enumerate(a_list_of_texts):
            tag = self.TAG_NAMES[maxcl[idx]]
            is_ood = True if cl_prob[idx] < self.ood_threshold else False
            if tag == "INSANE" and cl_prob[idx] > self.ood_threshold+0.002:
                is_ood = False
            self.logger.info("FAQ/OOD Classifier Predicted: text [%s] as [%s], prob.: %.4f, is ood?: %s",
                    a_text, tag, cl_prob[idx], "True" if is_ood else "False")
            ret_list.append((a_text, tag, cl_prob[idx], is_ood))

        return ret_list


