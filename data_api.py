"""
    data.go.kr의 API 연동을 위한 utility.
"""

import os
import re
import math
import datetime
import json
import logging
import asyncio
import aiohttp
import urllib.parse

from pytz import timezone

# Data.go.kr의 API 들은 응답에 시간이 걸린다. 최장 15초로 세팅하고, 보통 5초로 세팅해야 함.
# 3초로 세팅 시 timeout 오류가 날 것이다. aiohttp에서 따로 오류 핸들링을 무시하게 되므로,
# 바깥에서 exception의 오류 메시지가 잡히지 않는 것에 주의할 것.
HTTP_REQ_TIMEOUT_SECONDS=20

class async_DataKrAirQualityAPI:
    def __init__(self, logger):
        self.logger = logger
        self.SVC_URL = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"
        self.SVC_KEY = "HTMXgv2PPcDVbqEim0nJ7YCWYTLI/WrV1HMDf4imJOwQasgom6LGqzkXUBoWJeyo3+xjArf0mbxDVLilMAOdYQ=="

    async def query(self, where: str="대전"):
        request_params = { "serviceKey": self.SVC_KEY,
                         "pageNo": str(1),
                         "numOfRows": str(2000),
                         "returnType": "JSON",
                         "sidoName": where,
                         "ver": "1.0" }

        request_url = self.SVC_URL + "?" + urllib.parse.urlencode(request_params)
        items = {}
        totalCnt = 0
        # 5초로 제한
        timeout = aiohttp.ClientTimeout(total=HTTP_REQ_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(request_url) as resp:
                respraw = await resp.text()
                if resp.status == 200:
                    respbody = json.loads(respraw)
                    #print(respbody)
                    try:
                        ans = respbody["response"]["header"]["resultCode"]
                        if ans == "00":
                            # 정상. NORMAL_SERVICE.
                            items = respbody["response"]["body"]["items"]
                            # totalCnt > numOfRows일 경우 더 다운로드 받아야 함
                            totalCnt = respbody["response"]["body"]["totalCount"]
                        else:
                            msg = respbody["response"]["header"]["resultMsg"]
                            self.logger.error(f"Server Response Code: {ans}, message: {msg}")
                    except Exception as ex:
                        self.logger.warning("Data.Kr AirQuality API Output is NULL. fallback to default value.")
                        pass
                else:
                    pass

                return items, totalCnt

    def parse_dustgrade(self, items: list) -> str:
        """ 각 items를 iterate 하여 khaiGrade 수집 후, 가장 많은 것을 반환하도록 한다.
        수집 위치가 각기 다르고, 일부 기기는 점검/교정시 값이 나오지 않기 때문.
        한 시간 단위로 업데이트 된다고 생각하면 된다.
        """
        GRADE_POS = ["없음", "좋음", "보통", "나쁨", "매우나쁨"]
        grade_cnts = [0, 0, 0, 0, 0]

        for item in items:
            if "khaiGrade" in item and item["khaiGrade"] is not None:
                grade_cnts[int(item["khaiGrade"])] += 1

        self.logger.info(f"aggregated dustgrade - {grade_cnts[1:]}")
        max_v = max(grade_cnts)
        max_idx = grade_cnts.index(max_v)

        return GRADE_POS[max_idx]



class async_DataKrWeatherAPI:
    """
       data.go.kr의 기상청 단기 예보 API 요청 처리기
    """
    def __init__(self, logger):
        self.logger = logger
        # SVC_KEY는 외부 노출 금지. 2024-05-03 까지 유효함
        self.SVC_KEY = "HTMXgv2PPcDVbqEim0nJ7YCWYTLI/WrV1HMDf4imJOwQasgom6LGqzkXUBoWJeyo3+xjArf0mbxDVLilMAOdYQ=="
        # 단기/초단기 예보 조회 URL
        self.SVC_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
        # 단기 예보 API
        self.SHORT_PERIOD_ENDPOINT_API = "/getVilageFcst"
        # 초단기 예보 API
        self.ULTRA_SHORT_ENDPOINT_API = "/getUltraSrtFcst"
        # 초단기 실황 API (이것만 사용한다)
        self.ULTRA_CURRENT_ENDPOINT_API = "/getUltraSrtNcst"

    def get_day_and_time(self) -> (str, str):
        """ 프로필대화를 위한 정보로, 날짜와 시간 분류를 반환한다.
            day: [월요일, 화~목요일, 금요일]
            time: [오전 (9~11시), 점심 (11~14시), 오후 (14~17시), 퇴근 (17~18시)]
        """
        ret_weekday, ret_timescope = "", ""

        current_time = datetime.datetime.now(timezone('Asia/Seoul'))
        weekday = current_time.weekday()
        if weekday == 0:
            ret_weekday = "월요일"
        elif weekday >= 1 and weekday <= 3:
            ret_weekday = "화~목요일"
        elif weekday == 4:
            ret_weekday = "금요일"
        else:
            #ret_weekday = "휴일"      # 나와서는 안되지만, 일단은 넣는다.
            ret_weekday = "금요일"      # 나와서는 안되지만, 일단은 넣는다.

        if current_time.hour < 11:
            ret_timescope = "오전 (9~11시)"
        elif current_time.hour < 14:
            ret_timescope = "점심 (11~14시)"
        elif current_time.hour < 17:
            ret_timescope = "오후 (14~17시)"
        else:
            ret_timescope = "퇴근 (17~18시)"

        return ret_weekday, ret_timescope

    def create_request_url_for_current(self, nx: int=67, ny: int=101):
        """ 초단기 실황. basetime은 무조건 00으로 떨어짐. """
        base_date = datetime.date.today().strftime("%Y%m%d")
        current_time = datetime.datetime.now(timezone('Asia/Seoul'))
        if current_time.minute > 41:
            hrs = current_time.hour
        else:
            hrs = current_time.hour - 1

        if hrs < 0:
            hrs = 0

        base_time = '{0:02d}00'.format(hrs)
        request_params = { "ServiceKey": self.SVC_KEY,
                         "pageNo": str(1), "numOfRows": str(2000),
                         "dataType": "JSON",
                         "base_date": base_date, "base_time": base_time,
                         "nx": str(nx), "ny": str(ny) }

        request_url = self.SVC_URL + self.ULTRA_CURRENT_ENDPOINT_API + "?" + urllib.parse.urlencode(request_params)
        return request_url


    def create_request_url_for_ushort(self, nx: int=67, ny: int=101, base_date: str=""):
        """ 초단기 예보 """
        if base_date == "":
            base_date = datetime.date.today().strftime("%Y%m%d")

        current_time = datetime.datetime.now(timezone('Asia/Seoul'))
        if current_time.minute > 45:
            hrs = current_time.hour
        else:
            hrs = current_time.hour - 1

        base_time = '{0:02d}{1:02d}'.format(hrs, 30)

        request_params = { "ServiceKey": self.SVC_KEY,
                         "pageNo": str(1),
                         "numOfRows": str(2000),
                         "dataType": "JSON",
                         "base_date": base_date,
                         "base_time": base_time,
                         "nx": str(nx), "ny": str(ny) }

        request_url = self.SVC_URL + self.ULTRA_SHORT_ENDPOINT_API + "?" + urllib.parse.urlencode(request_params)
        return request_url

    def create_request_url(self, nx=67, ny=100, pageno=1, base_date="", base_time="0800"):
        """
        nx=67, ny=100은 대전광역시.
        nx=67, ny=101은 대전시 유성구.
        서울 - 60, 127. 부산 - 98, 76. 대구 - 89, 90. 인천 - 55, 124.
        광주 - 58, 74. 울산 - 102, 84. 세종 - 66, 103. 경기도 - 60, 120.
        강원도 - 73, 134. 충청북도 - 69, 107. 충청남도 - 68, 100. 전라북도 - 63, 89.
        전라남도 - 51, 67. 경상북도 - 89, 91. 경상남도 - 91, 77. 제주시 - 52, 38.
        이어도 - 28, 8.
        """
        if base_date == "":
            base_date = datetime.date.today().strftime("%Y%m%d")

        if base_time == "":
            base_time = "0800"

        request_params = { "ServiceKey": self.SVC_KEY,
                         "pageNo": str(pageno),
                         "numOfRows": str(2000),
                         "dataType": "JSON",
                         "base_date": base_date,
                         "base_time": base_time,
                         "nx": str(nx), "ny": str(ny) }

        request_url = self.SVC_URL + self.SHORT_PERIOD_ENDPOINT_API + "?" + urllib.parse.urlencode(request_params)
        return request_url

    async def query_current_weather(self, nx=67, ny=101):
        """ 초단기 예보 JSON 결과 수신 """
        req_url = self.create_request_url_for_current(nx, ny)
        print(req_url)
        items = {}
        totalCnt = 0
        timeout = aiohttp.ClientTimeout(total=HTTP_REQ_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(req_url) as resp:
                respraw = await resp.text()
                if resp.status == 200:
                    respbody = json.loads(respraw)
                    #print(respbody)
                    try:
                        ans = respbody["response"]["header"]["resultCode"]
                        if ans == "00":
                            # 정상. NORMAL_SERVICE.
                            items = respbody["response"]["body"]["items"]
                            # totalCnt > numOfRows일 경우 더 다운로드 받아야 함
                            totalCnt = items["totalCount"]
                        else:
                            msg = respbody["response"]["header"]["resultMsg"]
                            self.logger.error(f"Server Response Code: {ans}, message: {msg}")
                    except Exception as ex:
                        self.logger.warning("Data.kr Weather API #1 Output is NULL.")
                        pass
                else:
                    pass

                return items, totalCnt

    def parse_current(self, items: list) -> (str, str):
        """ 초단기 실황 데이터를 받아서, temperature와 weather를 반환.
            "temperature": [매우 추움, 추움, 선선함, 더움, 매우 더움]
            "weather" : [맑음, 흐림, 비, 눈]

            으로 반환한다. temperature는 풍속과 온도를 결합하여 체감 온도로 치환.

            체감온도는 여름철(5,6,7,8,9월), 겨울철(10,11,12,1,2,3,4월)로 계산한다.
            여름철 체감 온도 계산 공식: -0.2442 + 0.55399*습구온도 + 0.45535 * 기온 - 0.0022 * 습구온도^2 + 0.00278 * 습구온도 *기온 + 3.5
            습구온도 Stull 추정식: 기온 * ATAN[0.151977(상대습도+8.313659)^(1/2)] + ATAN(기온 + 상대습도) - ATAN(상대습도 - 1.67633) + 0.00391838 * 상대습도^(3/2) * ATAN(0.023101 * 상대습도) - 4.686035

            겨울철 체감 온도 계산 공식: 13.12 + 0.6215 * 기온 - 11.37 * 풍속^0.16 + 0.3965 * (풍속^0.16) * 기온

            계산공식 출처: 기상자료개방포털, https://data.kma.go.kr/climate/windChill/selectWindChillChart.do?pgmNo=111
        """
        ret_temp_str, ret_weather_str = "", ""
        # cel은 온도, reh는 습도, wsd는 풍속
        temp_cel, temp_reh, temp_wsd = 0, 0, 0
        month = -1
        latest_time = ""
        for item in items:
            if latest_time == "":
                latest_time = item["fcstTime"]
            elif latest_time != "" and latest_time != item["fcstTime"]:
                continue

            if month == -1:
                month = int(item["baseDate"][4:6])

            if item["category"] == "SKY":
                # 하늘 상태 코드: 맑음(1), 구름많은(3), 흐림(4)
                if ret_weather_str == "":
                    val = int(item["fcstValue"])
                    if val == 1:
                        ret_weather_str = "맑음"
                    elif val == 3 or val == 4:  # 3은 구름많음이나
                        ret_weather_str = "흐림"
            elif item["category"] == "PTY":
                # 초단기: 없음(0), 비(1), 비/눈(2), 눈(3), 소나기(4), 빗방울(5),
                # 빗방울눈날림(6), 눈날림(7)
                val = int(item["fcstValue"])
                if val == 1 or val == 4 or val == 5:
                    ret_weather_str = "비"
                elif val == 2 or val == 3 or val == 6 or val == 7:
                    ret_weather_str = "눈"
            elif item["category"] == "T1H":
                # 기온, celcius.
                temp_cel = float(item["fcstValue"])
            elif item["category"] == "REH":
                temp_reh = int(item["fcstValue"])
            elif item["category"] == "WSD":
                temp_wsd = float(item["fcstValue"])

        # 이제 온도 결정.
        try:
            if month >= 5 and month <= 9:
                # 여름철 계산
                # 여름철 체감 온도 계산 공식: -0.2442 + 0.55399*습구온도 + 0.45535 * 기온 - 0.0022 * 습구온도^2 + 0.00278 * 습구온도 *기온 + 3.5
                # 습구온도 Stull 추정식: 기온 * ATAN[0.151977(상대습도+8.313659)^(1/2)] + ATAN(기온 + 상대습도) - ATAN(상대습도 - 1.67633) + 0.00391838 * 상대습도^(3/2) * ATAN(0.023101 * 상대습도) - 4.686035
                tw = temp_cel * math.atan(0.151977 * math.pow((temp_reh + 8.313659), 0.5)) + \
                        math.atan(temp_cel + temp_reh) - math.atan(temp_reh - 1.67633) + \
                        0.00391838 * math.pow(temp_reh, 1.5) * math.atan(0.023101 * temp_reh) - 4.686035
                temp_corr = -0.2442 + 0.55399 * tw + 0.45535 * temp_cel - 0.0022 * math.pow(tw, 2) + \
                        0.00278 * tw * temp_cel + 3.5
                temp_corr = round(temp_corr, 1)

            else:
                # 겨울철 계산
                # 체감온도 = 13.12 + 0.6215 * 기온 - 11.37 * 풍속^0.16 + 0.3965 * (풍속^0.16) * 기온
                temp_corr = 13.12 + (0.6215 * temp_cel) - 11.37 * math.pow(temp_wsd, 0.16) + 0.3965 * math.pow(temp_wsd, 0.16) * temp_cel
                temp_corr = round(temp_corr, 1)
        except Exception as e:
            # 혹시나, 오류가 있으면 그냥 현재 온도를 사용
            self.logger.warning(f"exception occured: {e}")
            temp_corr = temp_cel
            pass

        self.logger.info("current temp: %.1f, calibrated sensible temp: %.1f", temp_cel, temp_corr)

        # 이제 온도 결정.
        # 22~12도 - 선선함, 12~5도 추움, 4도 이하 매우 추움
        # 23도~ 27도 더움, 28도 이상 매우더움
        if temp_corr >= 28:
            ret_temp_str = "매우 더움"
        elif temp_corr >= 23:
            ret_temp_str = "더움"
        elif temp_corr >= 13:
            ret_temp_str = "선선함"
        elif temp_corr >= 5:
            ret_temp_str = "추움"
        else:
            ret_temp_str = "매우 추움"

        return ret_temp_str, ret_weather_str


    async def query_ushort(self, base_date="", nx=67, ny=101):
        """ 초단기 예보 JSON 결과 수신 """
        req_url = self.create_request_url_for_ushort(nx, ny, base_date)
        #print(req_url)
        items = {}
        totalCnt = 0
        timeout = aiohttp.ClientTimeout(total=HTTP_REQ_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(req_url) as resp:
                respraw = await resp.text()
                if resp.status == 200:
                    respbody = json.loads(respraw)
                    #print(respbody)
                    try:
                        ans = respbody["response"]["header"]["resultCode"]
                        if ans == "00":
                            # 정상. NORMAL_SERVICE.
                            items = respbody["response"]["body"]["items"]
                            # totalCnt > numOfRows일 경우 더 다운로드 받아야 함
                            totalCnt = respbody["response"]["body"]["totalCount"]
                        else:
                            msg = respbody["response"]["header"]["resultMsg"]
                            self.logger.error(f"query_ushort(), Server Response Code: {ans}, message: {msg}, failed URL: [{req_url}]")
                    except Exception as ex:
                        self.logger.warning(f"Data.kr Weather API #2 exception occured. {ex}. request URL: [{req_url}]")
                        pass
                else:
                    pass

                return items, totalCnt


    async def query_shortperiod(self, base_date="", base_time="0800", nx=67, ny=100):
        """ 받은 아이템, totalCnt 수를 tuple로 반환.
            len(item) < totalCnt 일 경우 추가 요청 필요함 (pageno += 1)
        """
        req_url = self.create_request_url(nx, ny, 1, base_date, base_time)
        items = {}
        totalCnt = 0
        timeout = aiohttp.ClientTimeout(total=HTTP_REQ_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(req_url) as resp:
                respraw = await resp.text()
                if resp.status == 200:
                    respbody = json.loads(respraw)
                    #print(respbody)
                    try:
                        ans = respbody["response"]["header"]["resultCode"]
                        if ans == "00":
                            # 정상. NORMAL_SERVICE.
                            items = respbody["response"]["body"]["items"]
                            # totalCnt > numOfRows일 경우 더 다운로드 받아야 함
                            totalCnt = items["totalCount"]
                        else:
                            msg = respbody["response"]["header"]["resultMsg"]
                            self.logger.error(f"Server Response Code: {ans}, message: {msg}")
                    except Exception as ex:
                        self.logger.warning("Data.kr Weather API #3 Output is NULL.")
                        pass
                else:
                    pass

                return items, totalCnt


class TimerBasedLoopExecutor:
    def __init__(self, name, callback, interval_sec, logger,
                 execute_limit=None, execute_first: bool=True):
        """ callback은 partial을 사용해서 argument를 채우고 호출.

            name - 이름, callback - cb 메서드, interval_sec - 단위는 초.
            logger - getLogger() 인스턴스. execute_limit = None일 경우
        """
        self.name = name
        self.interval = interval_sec
        self.execute_limit = execute_limit
        self.execute_first = execute_first
        self.callback = callback
        self.logger = logger
        self._is_active = True
        self._task = asyncio.ensure_future(self._task_detail())

    async def _task_detail(self):
        exec_count = 0
        try:
            while self._is_active:
                if self.execute_first:
                    self.logger.debug("LoopExecutor: execute [%s]", self.name)
                    await self.callback()
                    self.logger.debug("LoopExecutor: wait [%d] secs", self.interval)
                    await asyncio.sleep(self.interval)
                else:
                    self.logger.debug("LoopExecutor: wait [%d] secs", self.interval)
                    await asyncio.sleep(self.interval)
                    self.logger.debug("LoopExecutor: execute [%s]", self.name)
                    await self.callback()

                if self.execute_limit is not None and exec_count > self.execute_limit:
                    break
        except Exception as e:
            self.logger.error(f"Exception occured in TimerBasedLoopExecutor - {e}")
            raise e

    def cancel(self):
        self.logger.info("task [%s] cancelled.", self.name)
        self._is_active = False
        self._task.cancel()


async def test():
    logger = logging.getLogger('')

    base_date = datetime.date.today().strftime("%Y%m%d")
    dkapi = async_DataKrWeatherAPI(logger)
    items, totalCnt = await dkapi.query_ushort()
    if "item" in items:
        temp, wea = dkapi.parse_current(items["item"])
        print(f"temperature - {temp}, weather - {wea}")
    else:
        logger.warning("item not found!")
        print(items)

    with open('test_out.json', 'w') as of:
        of.write(json.dumps(items, default=vars, sort_keys=True, indent=2, ensure_ascii=False))
        of.flush()
        of.close()

    """
    airapi = async_DataKrAirQualityAPI(logger)
    items, totalCnt = await airapi.query()
    output = airapi.parse_dustgrade(items)
    #print(items)
    print(output)
    """

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())
