import sys
import os
import re
import asyncio
import aiohttp


# wikiqa request timeout
HTTP_REQ_TIMEOUT_SECONDS = 3

class async_WikiQA:
    """ WikiQA의 비동기 버전. """
    def __init__(self, logger):
        self.WIKIQA_URL = "http://aiopen.etri.re.kr:8000/WikiQA"
        self.WIKIQA_ACCKEY = "0aa76349-6e0f-4aec-bb08-2736a3b35281"
        self.WIKIQA_TYPE = "hybridqa"       # irqa, 또는 hybridqa를 넣을 수 있음. 속도는 kbqa가 가장 빠름.
        self.logger = logger

    def get_request_json(self, question: str) -> dict:
        request_json = { "access_key": self.WIKIQA_ACCKEY,
                "argument": { "question": question,
                               "type": self.WIKIQA_TYPE }
                }
        return request_json

    async def query(self, question:str, threshold: float=0.5) -> (str, float):
        if len(question) == 0:
            return '', 0.0

        req_json = self.get_request_json(question)
        timeout = aiohttp.ClientTimeout(total=HTTP_REQ_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.WIKIQA_URL, json=req_json) as resp:
                    respbody = await resp.json()
                    if resp.status == 200:
                        #print(respbody)
                        try:
                            ans = respbody["return_object"]["WiKiInfo"]['AnswerInfo']
                            if len(ans) > 0:
                                if float(ans[0]['confidence']) >= threshold and len(ans[0]['answer']) > 1:
                                    print(f"answer found: {ans[0]['answer']}, confidence: {ans[0]['confidence']}, given threshold: {threshold}")
                                    return (ans[0]['answer'], ans[0]['confidence'])
                        except Exception as ex:
                            self.logger.info("WikiQA Output is NULL. it is safe behavior for Non-questions.")
                            pass
                    else:
                        pass

                    return '', 0.0
        except Exception as eox:
            self.logger.info(f"WikiQA Exception Occured, maybe timeout: {eox}")
            pass

        return '', 0.0


if __name__ == '__main__':
    import logging

    async def test():
        logger = logging.getLogger('')
        wqa = async_WikiQA(logger)
        while True:
            try:
                sys.stderr.write('>> ')
                sys.stderr.flush()
                input_msg = sys.stdin.readline().strip()
                if input_msg == "":
                    continue
                output, confidence = await wqa.query(input_msg)
                print(f'output: {output}, confidence: {confidence}')
            except Exception as e:
                print(f'Error!! : {e}')
                raise e

    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())
