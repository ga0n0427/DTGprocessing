"""
SearchBus_v.05.py
수정내역:
매직넘버 1 30

1. 1번 안 잡히는 문제 해결 
2. 마지막 번호 안 잡히는 문제 해결
3. 28번과 30번이 1초 차이나는 경우 해결 
"""

import numpy as np
import pandas as pd
import argparse
import os
import warnings
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 유틸리티 함수
# =============================================================================
def calDistance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    두 지리적 좌표 사이의 제곱 거리를 계산합니다.
    """
    return ((lat1 - lat2) ** 2) + ((lng1 - lng2) ** 2)

def parse_time(timestamp: str) -> datetime:
    """
    'YYMMDDHHMMSS' 형태의 타임스탬프 문자열을 datetime 객체로 변환합니다.
    """
    year = 2000 + int(timestamp[:2])
    month = int(timestamp[2:4])
    day = int(timestamp[4:6])
    hour = int(timestamp[6:8])
    minute = int(timestamp[8:10])
    second = int(timestamp[10:12])
    return datetime(year, month, day, hour, minute, second)

def getLimit(br: pd.DataFrame) -> float:
    """
    모든 버스 정류장 간의 거리를 계산하고,
    최소 거리의 절반 값을 반환합니다.
    """
    if br.empty or len(br) < 2:
        print("[경고] getLimit() - bus_route 데이터가 부족합니다:", br)
        return 0.00001  # 혹은 적당한 기본값으로 회피
    distance_list = []
    for i in range(len(br) - 1):
        for j in range(i + 1, len(br)):
            distance = calDistance(
                br.iloc[i]['LAT'], br.iloc[i]['LNG'],
                br.iloc[j]['LAT'], br.iloc[j]['LNG']
            )
            if distance != 0:
                distance_list.append(distance)

    distance = np.min(distance_list) / 2
    return distance

def prepare_directories(bus_path: str) -> str:
    """
    결과 파일을 저장할 디렉토리를 준비합니다.
    """
    path = bus_path.replace('orderbyNumber', 'sort_file').replace('.csv', '') + '/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# 2. 데이터 처리 관련 함수
# =============================================================================
def load_bus_route(route_path: str, route_name: str) -> pd.DataFrame:
    """
    지정된 노선 이름에 해당하는 버스 노선 데이터를 로드합니다.
    """
    bus_route = pd.read_csv(route_path, sep=',')
    bus_route = bus_route.astype({'BUS_ROUTE': 'str'})
    return bus_route[bus_route['BUS_ROUTE'] == route_name]

def process_bus_data(bus: pd.DataFrame) -> pd.DataFrame:
    """
    버스 데이터를 전처리(타임스탬프 파싱 및 정렬 등)합니다.
    """
    bus['Information_Occurrence'] = bus['Information_Occurrence'].apply(str)
    bus['Parsed_Date'] = bus['Information_Occurrence'].apply(parse_time)
    bus.sort_values('Parsed_Date', inplace=True)
    bus.reset_index(drop=True, inplace=True)
    bus['Formatted_Information_Occurrence'] = bus['Parsed_Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return bus

def process_one_file(file_path: str, route_path: str, route_name: str) -> None:
    """
    파일을 읽어와 결측치 제거 후,
    차량별로 분할하여 search 함수를 호출합니다.
    """
    try:
        b = pd.read_csv(file_path, sep=',', encoding='cp949')
    except UnicodeDecodeError:
        b = pd.read_csv(file_path, sep=',', encoding='euc-kr')

    b = b.dropna().reset_index(drop=True)

    # 차량별로 분리하여 search 호출
    car_names = set(b['Car_RegistrationNumber'].to_list())
    for car in car_names:
        bu = b[b['Car_RegistrationNumber'] == car].reset_index(drop=True)
        search(file_path, route_path, route_name, bu)


def calculate_mask(bus, bus_route, distance):
    """
    SciPy의 cdist로 모든 (버스 vs 정류장) 쌍의 제곱거리(sqeuclidean)를
    한 번에 계산한 후, 특정 임계값보다 작은 첫 번째 정류장의 순번을 mask에 기록합니다.
    """
    bus_lat = bus['LAT'].astype(float).to_numpy()  
    bus_lng = bus['LNG'].astype(float).to_numpy()
    route_lat = bus_route['LAT'].astype(float).to_numpy()
    route_lng = bus_route['LNG'].astype(float).to_numpy()

    # (N, 2), (M, 2) 모양의 좌표 배열
    points_bus = np.column_stack([bus_lat, bus_lng])      # shape (N, 2)
    points_route = np.column_stack([route_lat, route_lng])# shape (M, 2)

    # cdist: shape (N, M)
    # metric='sqeuclidean'으로 제곱거리 계산 (루트가 안 들어감)
    distances = cdist(points_bus, points_route, metric='sqeuclidean')

    # threshold보다 작은 것들
    valid = distances < distance

    # row(각 버스)에 대해 True인 정류장 중 가장 앞 인덱스
    mask_indices = np.argmax(valid, axis=1)

    # True가 전혀 없는 경우 처리
    no_valid = ~np.any(valid, axis=1)
    mask_indices[no_valid] = -1

    # 1-based
    mask = np.where(mask_indices >= 0, mask_indices + 1, 0)
    return mask

def filter_unique_mask(mask: np.ndarray) -> np.ndarray:
    """
    마스크에서 중복되지 않는 순서(첫 등장)만 필터링하여 순서대로 반환합니다.
    """
    m = mask[mask != 0]
    if len(m) == 0:
        return np.array([])
    _, unique_indices = np.unique(m, return_index=True)
    unique_mask = m[np.sort(unique_indices)]
    return unique_mask

def devide_up_down(bus_route: pd.DataFrame) -> int:
    """
    버스 노선이 상행/하행 혹은 분기점이 있는지 확인을 위한 예시 함수.
    특정 조건(연속 정류장 번호 불연속) 발생 시 해당 정류장의 MASK_SELECTED를 반환.

    ex) 1, 2, 3, ... 28 -> 1, 2, 3, ... 28 (A코스)
        1, 2, 3, ... 30 (B코스)
    """
    for i in range(1, len(bus_route)):
        if bus_route.loc[i, 'STOP_ORD'] != bus_route.loc[i - 1, 'STOP_ORD'] + 1:
            return bus_route.loc[i, 'MASK_SELECTED']  # 이 칼럼이 있을 경우
    # 기본 return
    return None

def check_time_diff(result_df: pd.DataFrame) -> bool:
    """
    주어진 result_df에서 시간 순으로 정렬하고,
    이전 정류장과의 시간 차(분)를 'TIME_GAP_MIN' 열에 기록한다.
    
    단, (27->28) 또는 (56->57) 구간일 때만 5분 이상인지 검사하여,
    5분 이상이면 True, 아니면 False를 반환.
    다른 구간은 검사하지 않음.
    """
    if result_df.empty or len(result_df) < 2:
        return False  # 비교 불가능
    
    temp_col = '_TMP_DT'
    
    # 'Information_Occurrence'를 datetime으로 변환 (이미 datetime이면 자동 변환됨)
    result_df[temp_col] = pd.to_datetime(result_df['Information_Occurrence'], errors='coerce')
    
    # 시간 순으로 정렬
    result_df.sort_values(by=temp_col, inplace=True, ignore_index=True)
    
    # 시간 차(분) 기록할 열 (첫 행은 비교 대상이 없으므로 0)
    result_df['TIME_GAP_MIN'] = 0.0

    over_5min = False
    for i in range(1, len(result_df)):
        prev_time = result_df.loc[i - 1, temp_col]
        curr_time = result_df.loc[i, temp_col]
        diff_minutes = (curr_time - prev_time).total_seconds() / 60.0 if pd.notnull(curr_time) and pd.notnull(prev_time) else 0
        
        # 현재 행의 TIME_GAP_MIN 갱신
        result_df.loc[i, 'TIME_GAP_MIN'] = diff_minutes
        
        prev_mask = result_df.loc[i - 1, 'MASK_SELECTED']
        curr_mask = result_df.loc[i, 'MASK_SELECTED']
        
        # (27->28) 또는 (56->57) 구간만 검사
        if ((prev_mask == 27 and curr_mask == 28) or
            (prev_mask == 56 and curr_mask == 57)):
            if diff_minutes >= 15:
                over_5min = True

    return over_5min



# =============================================================================
# 3. 결과 저장 및 마스킹 처리 함수
# =============================================================================
def save_result(result, path, route_name, bus, mask, num):
    # 결과 데이터를 저장합니다.
    if not result.empty:
        car_registration_number = bus['Car_RegistrationNumber'].iloc[0]
        result_filename = f"{path}{route_name}_{car_registration_number}_{num}.csv"
        result.to_csv(result_filename, index=False, encoding='cp949')

def save_result_visual(
    result: pd.DataFrame,
    path: str,
    route_name: str,
    bus: pd.DataFrame,
    mask: np.ndarray,
    num: int
) -> None:
    """
    결과 데이터를 저장하기 직전에,
    연속된 정류장 간 시간이 5분 이상 차이가 나는지
    (단, 28~29 사이 구간은 예외) 체크하고,
    해당되면 파일명을 print로 알려준다.
    """
    if not result.empty:
        car_registration_number = bus['Car_RegistrationNumber'].iloc[0]
        result_filename = f"{path}{route_name}_{car_registration_number}_{num}.csv"
        
        # (추가) 5분 이상 차이 체크
        if check_time_diff(result):
            print(f"[주의] 연속 정류장 간 5분 이상 차이가 있는 파일입니다: {result_filename}")
        
        # 결과 CSV 저장
        result.to_csv(result_filename, index=False, encoding='cp949')
        
        # 마스크 포함 데이터도 저장
        bus_with_mask = bus.copy()
        bus_with_mask['MASK'] = mask
        bus_filename = f"{path}{route_name}_{car_registration_number}_{num}_bus_data.csv"
        bus_with_mask.to_csv(bus_filename, index=False, encoding='cp949')


def process_mask(
    mask: np.ndarray,
    bus: pd.DataFrame,
    bus_route: pd.DataFrame,
    path: str,
    route_name: str,
    route_num: int
) -> None:
    """
    마스크를 시간 순으로 정렬하여 실제 정류장 도착(또는 근접) 지점을 필터링하고,
    일정 기준을 만족하면 결과를 저장합니다.
    """
    columns = ['BUS_ROUTE', 'STOP_ORD', 'STOP_NAME', 'LAT', 'LNG', 'STOP_ID',
               'Information_Occurrence', 'MASK_SELECTED']
    prev_time = None
    num = 0

    # 시간 순으로 정렬
    sorted_indices = bus['Parsed_Date'].argsort()
    sorted_mask = mask[sorted_indices]
    sorted_bus = bus.iloc[sorted_indices]

    last_stop = len(bus_route)   # 마지막 정류장 번호
    save_threshold = (last_stop / 2) - (last_stop / 4)  # 저장 기준값

    temp_result = pd.DataFrame(columns=columns)
    temp_before_ord = 1
    start_recording = False  # 1번 정류장 만나기 전 기록 시작 X

    last_1st_stop = None  
    last_28st_stop = None
    for i in range(len(sorted_mask)):
        current_time = sorted_bus['Parsed_Date'].iloc[i]
        limit_time = prev_time + timedelta(minutes=40) if prev_time else current_time

        # 1번 정류장을 만나기 전에는 기록 시작 안 함
        if not start_recording:
            if sorted_mask[i] == 1:
                start_recording = True
                prev_time = current_time  # 1번 정류장 만났을 때 시간 초기화
            else:
                continue

        # 40분 초과 시 기록 중단, temp_result 저장 여부 확인
        if current_time > limit_time:
            prev_time = current_time

            if len(temp_result) > save_threshold:
                num += 1
                save_result(temp_result, path, route_name, sorted_bus, sorted_mask, num)
            
            # 임시 결과 초기화
            temp_result = pd.DataFrame(columns=columns)
            temp_before_ord = 1

        current_ord = int(sorted_mask[i])
        if current_ord < 1:
            continue

        if current_ord == 1:
            last_1st_stop = (sorted_bus.iloc[i], current_time)

        if current_ord == (route_num - 1):
            last_28st_stop = (sorted_bus.iloc[i], current_time)

        # 정류장 번호 보정 여부 판단
        if current_ord > temp_before_ord and current_ord <= temp_before_ord + 3:
            corrected_ord = current_ord
            temp_before_ord = current_ord
        else:
            corrected_ord = (last_stop - (current_ord - 1))

            # 28번(또는 30번) 등을 맞춰주기 위한 예외 케이스
            if (current_ord == (route_num - 1)) and (corrected_ord == (route_num + 1)):
                corrected_ord = current_ord

            if corrected_ord >= len(bus_route):
                continue

            if corrected_ord > temp_before_ord and corrected_ord <= temp_before_ord + 3:
                temp_before_ord = corrected_ord
            else:
                continue

        # 2번 정류장 도착 전 1번 정류장 먼저 저장
        if corrected_ord >= 2 and last_1st_stop:
            time_diff = current_time - last_1st_stop[1]
            if time_diff <= timedelta(minutes=10):
                first_stop = bus_route.iloc[0]
                first_stop['LAT'] = last_1st_stop[0]['LAT']
                first_stop['LNG'] = last_1st_stop[0]['LNG']
                first_stop['Information_Occurrence'] = last_1st_stop[1]
                first_stop['MASK_SELECTED'] = 1
                temp_result = pd.concat([temp_result, first_stop.to_frame().T])
            last_1st_stop = None

        # 29번(등등) 이상 정류장 도착 전 28번(또는 30번) 정류장 먼저 저장
        if corrected_ord >= route_num and last_28st_stop:
            time_diff = current_time - last_28st_stop[1]
            if time_diff <= timedelta(minutes=10):
                last_stop_df = bus_route.iloc[28]  # 28번 인덱스
                last_stop_df['LAT'] = last_28st_stop[0]['LAT']
                last_stop_df['LNG'] = last_28st_stop[0]['LNG']
                last_stop_df['Information_Occurrence'] = last_28st_stop[1]
                last_stop_df['MASK_SELECTED'] = 29
                temp_result = pd.concat([temp_result, last_stop_df.to_frame().T])
            last_28st_stop = None

        if corrected_ord == route_num:
            continue

        # 현재 정류장 정보 기록
        current = bus_route.iloc[corrected_ord - 1]
        current['LAT'] = sorted_bus.iloc[i]['LAT']
        current['LNG'] = sorted_bus.iloc[i]['LNG']
        current['Information_Occurrence'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
        current['MASK_SELECTED'] = corrected_ord
        temp_result = pd.concat([temp_result, current.to_frame().T])
        
        # 마지막 정류장 도착 시 초기화
        if corrected_ord == last_stop:
            last_1st_stop = None
            temp_before_ord = 1
            continue

        # prev_time 업데이트
        prev_time = current_time

    # 루프 종료 후 마지막 temp_result 저장여부 확인
    if len(temp_result) > save_threshold:
        num += 1
        save_result(temp_result, path, route_name, sorted_bus, sorted_mask, num)


# =============================================================================
# 4. 주요 로직(검색) 함수
# =============================================================================
def search(bus_path: str, route_path: str, route_name: str, bus: pd.DataFrame) -> None:
    """
    주어진 버스 데이터와 노선 데이터를 기반으로
    1) 디렉토리 준비
    2) 노선 로드
    3) 데이터 전처리
    4) 마스킹 계산
    5) 마스킹 확인 후 처리 및 결과 저장
    일련의 단계를 수행합니다.
    """
    path = prepare_directories(bus_path)
    bus_route = load_bus_route(route_path, route_name)
    distance = getLimit(bus_route)

    # 버스 데이터 전처리
    bus = process_bus_data(bus)
    # 마스크 계산
    mask = calculate_mask(bus, bus_route, distance)

    # 마스크 필터링(첫 등장만 추출)
    unique_mask = filter_unique_mask(mask)
    #print("Unique mask:", unique_mask)

    # 유의미한 마스크가 아니면 종료
    if len(unique_mask) < 10:
        return

    # 노선 분기점(예: 28번, 30번) 판단
    route_num = devide_up_down(bus_route)

    # 마스크를 활용해 실제 정류장 도착 지점 저장
    process_mask(mask, bus, bus_route, path, route_name, route_num)


# =============================================================================
# 5. 메인 실행부
# =============================================================================
def main():
    """
    메인 함수: bus_path 내 모든 파일을 대상으로
    4개 스레드로 병렬 처리합니다.
    """
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--bus_path', required=True, help='버스 데이터 폴더 경로')
    parser.add_argument('--route_path', required=True, help='노선 데이터 파일 경로')
    parser.add_argument('--route_name', required=True, help='찾고자 하는 노선 이름')

    
    args = parser.parse_args()
    bus_path = args.bus_path
    route_path = args.route_path
    route_name = args.route_name
    
    # bus_path에 있는 모든 파일 경로 수집
    file_list = []
    for (path, dir, files) in os.walk(bus_path):
        for filename in files:
            file_path = os.path.join(path, filename)
            file_list.append(file_path)
    # 수집된 파일들을 4개 스레드로 병렬 처리
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_one_file, file_path, route_path, route_name)
            for file_path in file_list
        ]

    # (옵션) 모든 스레드 작업 완료 대기
    for f in futures:
        f.result()


if __name__ == '__main__':
    main()
