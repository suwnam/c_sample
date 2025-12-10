"""
   SQLite FTS5 기반 FAQ 매칭 모듈.

   Copyright (C) 2022. Jong-hun Shin. ETRI LIRS.
"""

import random
import re
import json

import sqlite3

# install mecab-ko and mecab-ko-dic w/conda.
import mecab
import pandas as pd

from nltk.metrics import distance
# install jamo package for python 3.
from jamo import h2j, j2hcj

from text_util import (refine_fts5_query, refine_faq_text)


class BM25FaqSearch:
    def __init__(self, faq_db_filename, logger):
        self.mecab_instance = mecab.MeCab()
        self.logger = logger

        self.eomi_bigram_blacklist = set(['거든', '지요', '단다', '었다', '었지', '네요', '하다', '막다',
                                          '많다', '적다', '좋다', '럽다', '였다', '렇다', '쁘다', '지네',
                                          '거다',])
        self.eomi_trigram_blacklist = set(['중이야', '하세요', '합시다', '했구요', '었구요', '있구요',
                                           '먹었어','막았어','빨았어','해봐요','자구요','어봐요',' 왔어',
                                           '좋았어', '싫었어','복했어','고 해', '는데요', ' 덥네', ' 춥네',
                                           ' 덥다', ' 춥다', '쪽이야', '쪽이다', '쪽인걸', '쪽이여',
                                           '것이다', '야겠어', ' 갔다', ' 싶다', '프잖아', '혼났어',
                                           '싶었어', ' 슬퍼', '슬프다', '버스야', '해보자', '래보자'])

        self.allow_surface = set(['지금', '몇', '오늘'])
        # UNA - unknown, VX - 보조용언
        self.pos_blacklist = set(['MAG', 'MAJ',
                                  'SF', 'IC', 'UNA', 'EP', 'VX', 'MM',
                                  'ETM', 'EF', 'EC',
                                  'JX', 'JKO', 'JKS', 'JKB', 'JKC', 'JKG', 'JKV', 'JKQ',
                                 ])
        self.query_filter = set(['OR', '(', 'AND', ')', '[(', ')]',])

        self.and_pos_set = set(['SN', 'NNP'])
        # 여기에는 SN, N* 품사만 들어간다.
        self.haeksim_words = set(['어디_NP', '언제_NP'])

        # 정규표현식
        self.morph_pos_split_re = re.compile("^(.*)/(.*)/(.*)$")
        self.whitespace_re = re.compile(r'\s+')

        self.logger.info("now start parsing FAQ DB.")

        self.base_db = self.read_excel_to_db(faq_db_filename, self.mecab_instance)

        self.fts5_db = sqlite3.connect(":memory:")
        cur = self.fts5_db.cursor()
        cur.execute("""CREATE VIRTUAL TABLE faqdb
                       USING fts5(ques, seg_ques, category,
                       content_filename UNINDEXED, responses UNINDEXED, nospace_ques, chartokenized_ques, tokenize="unicode61 tokenchars '-_&:'");""")
        cur.executemany("""INSERT INTO faqdb (ques, seg_ques, category, content_filename, responses, nospace_ques, chartokenized_ques)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""", self.base_db)
        self.fts5_db.commit()
        self.logger.info("FAQ DB Index Initialized.")

    def gen_query_string(self, input_str, is_index=True, print_query=False) -> str:
        if self.mecab_instance is None:
            return input_str

        or_tokens = []
        and_tokens = []

        # 형태소-품사 구분자
        mp_sep = '_'

        morphout = self.mecab_instance.parse(input_str)
        #display(morphout)
        # blacklisting: MAG(문장부사/양상부사), JX(주격조사), EC(어미), JKB(목적격조사)
        before_pos = ''
        final_query = ''

        # 수사(SN)은 AND 토큰으로, 수사 뒤 명사도 AND 토큰으로 넣을 것
        # 나머지는 다 OR 토큰으로 넣는다. 우선은.

        for a_morph, elem in morphout:
            if elem.expression is not None:
                compound = elem.expression.split('+')
                for in_morph in compound:
                    splits = self.morph_pos_split_re.match(in_morph)
                    inner_mor, inner_pos = splits.groups()[0], splits.groups()[1]
                    if inner_pos == 'VA':
                        inner_pos = 'VV'  # 형용사는 VV로 통일. 있_VV, 있_VA 차이 때문.
                    if splits.groups()[1] not in self.pos_blacklist or splits.groups()[0] in self.allow_surface:
                        a_tok = inner_mor + mp_sep + inner_pos
                        if splits.groups()[1] == 'SN' or \
                        (before_pos == 'SN' and splits.groups()[1][0] == 'N') or \
                        a_tok in self.haeksim_words:
                            and_tokens.append(a_tok)
                        else:
                            or_tokens.append(a_tok)
                        before_pos = splits.groups()[1]
            elif elem.pos not in self.pos_blacklist or a_morph in self.allow_surface:
                a_pos = elem.pos
                if a_pos == 'VA':
                    a_pos = 'VV'
                a_tok = a_morph + mp_sep + a_pos

                # 일부 결합 형태소 일반화
                if a_morph == '어딨' and a_pos == 'VV':
                    a_tok = '어디_NP 이_VCP'

                if (before_pos == 'SN' and a_pos[0] == 'N') or a_pos == 'SN' or \
                a_tok in self.haeksim_words or a_tok == 'ETRI':
                    and_tokens.append(a_tok)
                else:
                    or_tokens.append(a_tok)
                before_pos = elem.pos
                #print(a_morph, elem.pos)

        if is_index == True:
            final_query = ' '.join(and_tokens + or_tokens)
        else:
            if len(and_tokens) > 0:
                final_query += '( ' + ' '.join(and_tokens) + ' )'

            if len(or_tokens) > 0:
                if final_query != '':
                    final_query += ' AND '
                final_query += '( ' + ' OR '.join(or_tokens) + ' )'

        final_query = final_query.strip()

        if print_query:
            self.logger.info(f'Final Query: [{final_query}]')
        return final_query

    def read_excel_to_db(self, filename, mecab_obj):
        df = pd.read_excel(filename)

        default_sol = {}
        exact_dbs = []

        for idx, a_row in df.iterrows():
            solkey = a_row[1] + '/' + a_row[2]
            a_text = a_row[3]
            a_text = refine_faq_text(a_text)

            nospace_text = self.whitespace_re.sub("", a_text)
            chartokenized_text = ' '.join([char for char in nospace_text])

            keyword_ext_base = a_row[2]
            self.haeksim_words.update([x for x in self.gen_query_string(keyword_ext_base).split(' ') if re.search(r"(_SN|_NNG|_NNP)", x) is not None])

            if mecab_obj is not None:
                #segmented = ' '.join(mecab_obj.morphs(a_text))
                segmented = self.gen_query_string(a_text, True)
            else:
                self.logger.warning("MeCab instance not activated. segmented text will be same as orignal text.")
                segmented = a_text

            if not pd.isnull(a_row[5]):
                # 값이 있는 경우
                content_filename = '' if pd.isnull(a_row[4]) else a_row[4]
                resp_text = []
                for ridx in range(5, 10, 1):
                    if not pd.isnull(a_row[ridx]):
                        resp_text.append(a_row[ridx])
                    else:
                        break
                assert len(resp_text) > 0

                if solkey not in default_sol:
                    default_sol[solkey] = (content_filename, resp_text)

                exact_dbs.append((a_text, segmented, solkey, content_filename, json.dumps(resp_text, ensure_ascii=False), nospace_text, chartokenized_text))
            else:
                # 값이 없는 경우
                if solkey not in default_sol:
                    print('WARNING: solution must be appeared before empty element. omitting this question. idx: {0}, key: {1}, question: {2}'.format(idx, solkey, a_row[3]))
                else:
                    sol = default_sol[solkey]

                exact_dbs.append((a_text, segmented, solkey, sol[0], json.dumps(sol[1], ensure_ascii=False), nospace_text, chartokenized_text))

        self.logger.info(f"Exact Matching DB loaded, questions: {len(exact_dbs)}")
        #self.logger.debug(f"Core Content Word list: {list(self.haeksim_words)}")

        #         0               1              2            3                     4
        # list[question, segmented_question, category, content_filename, jsonified responses(list)]
        return exact_dbs

    def phase1_exact_matching(self, given_text: str) -> (str, str, str):
        # RESPONSE_TEXT 중 하나, CONTENT_FILENAME을 반환
        q = refine_fts5_query(given_text)
        nspace_q = self.whitespace_re.sub("", q)
        if len(q) < 3:
            self.logger.info("FAQ Phase1 reject, less than 3 chars.")
            return ("", "", "")

        cur = self.fts5_db.cursor()
        res = cur.execute(f"SELECT *, bm25(faqdb, 10.0, 5.0) FROM faqdb WHERE ques MATCH '{q}' or nospace_ques MATCH '{nspace_q}' ORDER BY bm25(faqdb, 10.0, 5.0) limit 3;").fetchall()
        if len(res) > 0:
            self.logger.info("FAQ Phase1 search - %d found.", len(res))
            self.logger.info("FAQ Phase1 Top-1 result: %s", str(res[0]))
            picked = res[0]

            # 여기에서도 전체 매칭이 50% 미만이면 reject.
            qs = set(self.whitespace_re.split(q))
            ms = set(picked[0].split(' '))
            intersect = list(qs & ms)
            self.logger.info(f"QS: {qs}, MS: {ms}, INTERSECT: {intersect}")
            # 여기는 ms 길이의 절반 이상은 되어야 한다.
            if q.count(' ') == 0 or len(q) < 6:
                # EXACT 매칭이 아니면 거절. nospace로 정규화 된 것을 비교하여 평가함.
                if picked[5] != nspace_q:
                    self.logger.info("FAQ Phase1 Rejected, no spaces or less than 6 chars, but Not Exact: %s", str(res[0]))
                    return ("", "", "")
                else:
                    responses = json.loads(picked[4])
            elif len(intersect) >= int(len(ms) * 0.5):
                responses = json.loads(picked[4])
            else:
                self.logger.info("FAQ Phase1 Rejected, Top-1 query intersection count: %d, top-1 message token count: %d",
                                 len(intersect), len(ms))
                return ("", "", "")

            responses = json.loads(picked[4])
            return (random.choice(responses), picked[3], picked[2])
        else:
            self.logger.info("FAQ Phase1 search - empty")
            return ("", "", "")

    def phase1p5_jamo_fuzzy_matching(self, given_text: str) -> (str, str, str):
        # jamo fuzzy matching.
        q = refine_fts5_query(given_text)
        nospace_q = self.whitespace_re.sub("", q)
        q = ' OR '.join([char for char in nospace_q])
        cur = self.fts5_db.cursor()
        res = cur.execute(f"SELECT *, bm25(faqdb, 10.0, 5.0) FROM faqdb WHERE chartokenized_ques MATCH '{q}' ORDER BY bm25(faqdb, 10.0, 5.0) limit 10;").fetchall()
        if len(res) > 0:
            self.logger.info("FAQ Phase1.5 search - %d found. calculate jamo-based edit distance.", len(res))
            for item in res:
                jdistance = distance.edit_distance(j2hcj(h2j(item[5])), j2hcj(h2j(nospace_q)))
                # only 1-jamo difference will accepted.
                if jdistance < 2:
                    self.logger.info("distance < 2 element found: original query: %s, matched(whitespace removed): %s", given_text, item[5])
                    responses = json.loads(item[4])
                    return (random.choice(responses), item[3], item[2])

        return ("", "", "")

    def phase2_fts5_and_matching(self, given_text: str, category: str) -> (str, str, str):
        # 2단계, 형분 후 AND 매칭. RESPONSE_TEXT 중 하나, CONTENT_FILENAME을 반환
        # 필요에 따라 조사, 어미를 제거해야 함.
        #q = ' '.join(self.mecab_instance.morphs(refine_fts5_query(given_text)))
        q = self.gen_query_string(refine_fts5_query(given_text), True, True)
        res = []
        # 우선은 길이만 보고 결정한다.
        if q.count(' ') < 2 or len(q) < 4 or given_text.count(' ') < 1:
            self.logger.info("FAQ Phase2, length insufficient to process query. q: %s", q)
            return ("", "", "")
        elif q.count('_NNG') == 0 and q.count('_NNP') == 0 and q.count('_SL') == 0:
            self.logger.info("FAQ Phase2, reject Empty Noun(NNG, NNP, SL) Query. q: [%s]", q)
            return ("", "", "")
        elif category == 'OOD':
            self.logger.info("FAQ Phase2, Reject OOD.")
            return ("", "", "")

        try:
            cur = self.fts5_db.cursor()
            if category == 'INSANE':
                res = cur.execute(f"SELECT *, bm25(faqdb, 10.0, 5.0) FROM faqdb WHERE seg_ques MATCH '{q}' ORDER BY bm25(faqdb, 10.0, 5.0) limit 3;").fetchall()
            elif category != '':
                res = cur.execute(f"SELECT *, bm25(faqdb, 10.0, 5.0) FROM faqdb WHERE category MATCH '{category}' AND seg_ques MATCH '{q}' ORDER BY bm25(faqdb, 10.0, 5.0) limit 3;").fetchall()
        except Exception as e:
            self.logger.warning('FAQ Phase2 exception found, error: [%s]', e)
            pass

        if len(res) > 0:
            self.logger.info("FAQ Phase2 search - %d found.", len(res))
            self.logger.info("FAQ Phase2 Top-3 result: %s", str(res[0:3]))
            picked = res[0]
            responses = json.loads(picked[4])
            return (random.choice(responses), picked[3], picked[2])
        else:
            self.logger.info("FAQ Phase2 search - empty")
            return ("", "", "")

    def phase3_fts5_extended_matching(self, given_text: str, category: str) -> (str, str, str):
        # 3단계, 형분 후 OR 매칭. RESPONSE_TEXT 중 하나, CONTENT_FILENAME을 반환
        # 필요에 따라 조사, 어미를 제거하고, NEAR() 로 query를 redefine 할 필요가 있음
        # 대신 category를 입력하게 해서 맞춤
        #q = ' OR '.join(self.mecab_instance.morphs(refine_fts5_query(given_text)))
        if category == '':
            return ("", "", "")

        q = self.gen_query_string(refine_fts5_query(given_text), False, True)
        res = []
        if len(q) < 8 or q.count('_') < 2 or (q.count('AND') == 0 and q.count('OR') >= 1):
            self.logger.info("FAQ Phase3, length insufficient to process query. q: %s", q)
            return ("", "", "")
        elif category == 'OOD':
            self.logger.info("FAQ Phase3, Reject OOD.")
            return ("", "", "")

        try:
            cur = self.fts5_db.cursor()
            if category == 'INSANE':
                res = cur.execute(f"SELECT *, bm25(faqdb, 10.0, 5.0) FROM faqdb WHERE seg_ques MATCH '{q}' ORDER BY bm25(faqdb, 10.0, 5.0) limit 3;").fetchall()
            else:
                res = cur.execute(f"SELECT *, bm25(faqdb, 10.0, 5.0) FROM faqdb WHERE category MATCH '{category}' AND seg_ques MATCH '{q}' ORDER BY bm25(faqdb, 10.0, 5.0) limit 3;").fetchall()
        except Exception as e:
            self.logger.warning('FAQ Phase3 exception found, error: [%s]', e)
            pass


        if len(res) > 0:
            self.logger.info("FAQ Phase3 search - %d found.", len(res))
            picked = res[0]

            # 검증해야 한다. OR 매칭이기 때문에, INTERSECT의 수가 전체의 50% 이상은 되어야 함.
            qs = set(self.whitespace_re.split(q))
            #self.logger.warning(f"QS #1 -> {qs}")
            qs = set(qs - self.query_filter)
            #self.logger.warning(f"QS #2 -> {qs}")
            ms = set(picked[1].split(' '))
            intersect = list(qs & ms)
            self.logger.warning(f"QS: {qs}, MS: {ms}, INTERSECT: {intersect}")

            if len(intersect) >= int(len(qs) * 0.75) and len(intersect) >= int(len(ms) * 0.75):
                responses = json.loads(picked[4])
            else:
                self.logger.info("FAQ Phase3 Rejected, Top-1 query intersection count: %d, top-1 message token count: %d, query token count: %d",
                                 len(intersect), len(ms), len(qs))
                return ("", "", "")

            self.logger.info("FAQ Phase3 Accepted, Top-3 result: %s", str(res[0:3]))

            return (random.choice(responses), picked[3], picked[2])
        else:
            self.logger.info("FAQ Phase3 search - empty")
            return ("", "", "")
        return

    def faq_query(self, given_text: str, category:str = '') -> (str, str, str):
        r, c, d = self.phase1_exact_matching(given_text)
        if r != "":
            return (r, c, d)

        r, c, d = self.phase1p5_jamo_fuzzy_matching(given_text)
        if r != "":
            return (r, c, d)

        # 일부 문미로 끝나는 평서문은 아예 매칭되지 않도록 막는다.
        if given_text[-2:] in self.eomi_bigram_blacklist:
            self.logger.info("FAQ Rejected Phase 2/3 by eomi: [%s]", given_text[-2:])
            return ("", "", "")
        elif given_text[-3:] in self.eomi_trigram_blacklist:
            self.logger.info("FAQ Rejected Phase 2/3 by eomi: [%s]", given_text[-2:])
            return ("", "", "")
        elif given_text.find("어서") != -1 and given_text[-4:] == '고 있어' or given_text[-3:] == '고있어':
            self.logger.info("FAQ Rejected Phase 2/3 by eomi, 어서-*-고-있어: [%s]", given_text)
            return ("", "", "")

        try:
            res = self.mecab_instance.parse(given_text)
            res = res[-3:]
            if len(res) == 3:
                if res[0][1].pos == 'NNG' and (res[1][1].pos[0] == 'V' and res[1][0] != '하' and res[1][1].pos.find("EP") == -1 ) and res[2][1].pos == 'EC':
                    self.logger.info("FAQ Rejected Phase 2/3 by MA pattern - NNG+V+EC: [%s]", given_text)
                    return ("", "", "")
            elif len(res) >= 2:
                res = res[-2:]
                if (res[0][1].pos == 'VA' and res[1][0] == '다') or (res[0][1].pos == 'NNG' and (res[1][0] == '다' or res[1][0] == '야')):
                    self.logger.info("FAQ Rejected Phase 2/3 by eomi - VA+다: [%s]", given_text[-2:])
                    return ("", "", "")
        except Exception as e:
            self.logger.error('faq_query(), exception occured: %s', e)
            pass

        r, c, d = self.phase2_fts5_and_matching(given_text, category)
        if r != "":
            return (r, c, d)
        # recall이 확 올라가지만 앞에 필터링을 해야 한다
        r, c, d = self.phase3_fts5_extended_matching(given_text, category)
        if r != "":
            return (r, c, d)
        return ("", "", "")


#bm25s = BM25FaqSearch('./faqdata/faq_exact_db_v220518.xlsx', logger)
