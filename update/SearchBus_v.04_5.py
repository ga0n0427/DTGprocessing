"""
SearchBus_v.04_5
수정내역: 
1. 보정값 적용 확인
2. 시간 단위 수정하여 하행도 잡히게 변경 
3. 초단위로 저장되게 수정
"""


import numpy as np
import pandas as pd
import argparse
import os
import warnings
from datetime import timedelta, datetime

warnings.filterwarnings('ignore')

def calDistance(lat1, lng1, lat2, lng2):
    # 두 지리적 좌표 사이의 제곱 거리를 계산합니다.
    return ((lat1 - lat2) ** 2) + ((lng1 - lng2) ** 2)

def getLimit(br):
    distance_list = []

    # 모든 버스 정류장 간의 거리를 계산합니다.
    for i in range(len(br) - 1):
        for j in range(i + 1, len(br)):
            distance = calDistance(br.iloc[i]['LAT'], br.iloc[i]['LNG'], br.iloc[j]['LAT'], br.iloc[j]['LNG'])
            if distance != 0:
                distance_list.append(distance)

    # 최소 거리의 절반 값을 반환합니다.
    distance = np.min(distance_list) / 2 
    return distance

def parse_time(timestamp):
    # 타임스탬프 문자열을 datetime 객체로 변환합니다.
    year = 2000 + int(timestamp[:2])
    month = int(timestamp[2:4])
    day = int(timestamp[4:6])
    hour = int(timestamp[6:8])
    minute = int(timestamp[8:10])
    second = int(timestamp[10:12])
    return datetime(year, month, day, hour, minute, second)

def prepare_directories(bus_path):
    # 결과 파일을 저장할 디렉토리를 준비합니다.
    path = bus_path.replace('orderbyNumber', 'sort_file').replace('.csv', '') + '/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def load_bus_route(route_path, route_name):
    # 지정된 노선 이름에 해당하는 버스 노선 데이터를 로드합니다.
    bus_route = pd.read_csv(route_path, sep=',', encoding='cp949')
    bus_route = bus_route.astype({'BUS_ROUTE': 'str'})
    return bus_route[bus_route['BUS_ROUTE'] == route_name]

def process_bus_data(bus):
    # 버스 데이터를 전처리합니다.
    bus['Information_Occurrence'] = bus['Information_Occurrence'].apply(str)
    bus['Parsed_Date'] = bus['Information_Occurrence'].apply(parse_time)
    bus.sort_values('Parsed_Date', inplace=True)
    bus.reset_index(drop=True, inplace=True)
    return bus

def calculate_mask(bus, bus_route, distance):
    # 버스 위치와 가장 가까운 정류소를 찾기 위해 마스크를 계산합니다.
    mask = np.zeros(len(bus), dtype=int)
    bus_lat = bus['LAT'].astype(float).to_numpy()
    bus_lng = bus['LNG'].astype(float).to_numpy()

    for i in range(len(bus)):
        for j in range(len(bus_route)):
            rlat = bus_route.iloc[j]['LAT']
            rlng = bus_route.iloc[j]['LNG']

            # 정류장과 버스의 거리 차이가 특정 값보다 가깝다면 mask에 정류장의 순번을 기록합니다.
            if calDistance(bus_lat[i], bus_lng[i], rlat, rlng) < distance:
                mask[i] = j + 1
                break
    return mask

def filter_unique_mask(mask):
    # 마스크에서 유일한 값을 필터링하여 반환합니다.
    m = mask[mask != 0]
    try:
        _, unique_indices = np.unique(m, return_index=True)
        unique_mask = m[np.sort(unique_indices)]
    except IndexError:
        unique_mask = []
    return unique_mask

def save_result(result, path, route_name, bus, num):
    # 결과 데이터를 저장합니다.
    if not result.empty:
        # 'Information_Occurrence' 열을 datetime으로 변환 후 포맷팅
        result['Information_Occurrence'] = pd.to_datetime(result['Information_Occurrence'])
        result['Information_Occurrence'] = result['Information_Occurrence'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        
        car_registration_number = bus['Car_RegistrationNumber'].iloc[0]
        result_filename = f"{path}{route_name}_{car_registration_number}_{num}.csv"
        result.to_csv(result_filename, index=False, encoding='cp949')


        
def process_mask(mask, bus, bus_route, path, route_name):
    # 마스크를 처리하여 결과 데이터를 생성합니다.
    columns = ['BUS_ROUTE', 'STOP_ORD', 'STOP_NAME', 'LAT', 'LNG', 'STOP_ID', 'Information_Occurrence']
    result = pd.DataFrame(columns=columns)
    prev_time = None
    num = 0

    # 시간 순으로 정렬합니다.
    sorted_indices = bus['Parsed_Date'].argsort()
    sorted_mask = mask[sorted_indices]
    sorted_bus = bus.iloc[sorted_indices]
    
    last_stop = len(bus_route)  # 마지막 정류장 번호
    save_threshold = (last_stop / 2) - (last_stop / 4)  # 저장 기준 값

    temp_result = pd.DataFrame(columns=columns)
    temp_before_ord = 1  # 1번 정류장부터 시작
    start_recording = False  # 1번 또는 28번 정류장을 만나기 전까지 기록 시작하지 않음

    for i in range(len(sorted_mask)):
        current_time = sorted_bus['Parsed_Date'].iloc[i]
        limit_time = prev_time + timedelta(minutes=40) if prev_time else current_time
        
        # 1번 또는 28번 정류장을 만나기 전에는 기록을 시작하지 않음
        if not start_recording:
            if sorted_mask[i] == 1 or sorted_mask[i] == 28:
                start_recording = True
                prev_time = current_time  # 1번 또는 28번 정류장을 만났을 때 prev_time 초기화
            else:
                continue  # 1번 또는 28번 정류장을 만나기 전까지는 다음 데이터로 넘어감

        if current_time > limit_time:
            # 40분 초과 시 이전 시간 초기화
            prev_time = current_time

            # 임시 결과가 저장 기준을 넘는지 확인
            if len(temp_result) > save_threshold:
                result = pd.concat([result, temp_result], ignore_index=True)
                num += 1
                save_result(result, path, route_name, sorted_bus, num)
            # 임시 결과 초기화
            temp_result = pd.DataFrame(columns=columns)
            temp_before_ord = 1  # 1번 정류장부터 시작

        current_ord = int(sorted_mask[i])

        if current_ord < 1:
            continue  # 유효하지 않은 정류장 번호는 건너뜁니다.

        if current_ord > temp_before_ord and current_ord <= temp_before_ord + 3:
            # 현재 정류장이 이전 정류장보다 크고 3까지 클 때의 범위의 수일 경우 그대로 사용
            corrected_ord = current_ord  # 보정되지 않은 정류장 번호 사용
            temp_before_ord = current_ord
        else:
            # 보정값을 계산
            corrected_ord = (last_stop - (current_ord - 1))
            if corrected_ord >= len(bus_route):
                continue  # 보정값이 정류장 인덱스를 벗어날 경우 다음 데이터로 넘김

            if corrected_ord > temp_before_ord and corrected_ord <= temp_before_ord + 3:
                # 보정값이 이전 정류장보다 크고 3까지 클 때의 범위의 수일 경우 사용
                temp_before_ord = corrected_ord
            else:
                continue  # 조건을 모두 만족하지 않으면 다음 데이터로 넘김

        # 임시 결과에 기록
        current = bus_route.iloc[corrected_ord - 1]
        current['LAT'] = sorted_bus.iloc[i]['LAT']
        current['LNG'] = sorted_bus.iloc[i]['LNG']
        current['Information_Occurrence'] = current_time
        temp_result = pd.concat([temp_result, current.to_frame().T])
        
        # prev_time을 current_time으로 업데이트
        prev_time = current_time

    # 마지막 임시 결과가 저장 기준을 넘는지 확인
    if len(temp_result) > save_threshold:
        result = pd.concat([result, temp_result], ignore_index=True)
        num += 1
        save_result(result, path, route_name, sorted_bus, num)





def search(bus_path, route_path, route_name, bus):
    # 주어진 버스 데이터와 노선 데이터를 기반으로 검색을 수행합니다.
    path = prepare_directories(bus_path)
    bus_route = load_bus_route(route_path, route_name)
    distance = getLimit(bus_route)
    bus = process_bus_data(bus)
    mask = calculate_mask(bus, bus_route, distance)
    
    unique_mask = filter_unique_mask(mask)
    print(unique_mask)
    if len(unique_mask) < 10:
        return 0

    if 'H' in bus.columns:
        return
    
    process_mask(mask, bus, bus_route, path, route_name)

def main():
    # 메인 함수로, 명령줄 인수를 처리하고 전체 프로세스를 실행합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('bus_path', help='버스 데이터 폴더 경로')
    parser.add_argument('route_path', help='노선 데이터 파일 경로')
    parser.add_argument('route_name', help='찾고자 하는 노선 이름. -1을 입력하면 모든 노선을 검색합니다.')
    
    args = parser.parse_args()

    for (path, dir, files) in os.walk(args.bus_path):
        for filename in files:
            b = pd.read_csv(os.path.join(path, filename), sep=',', encoding='cp949')
            if 'H' in b.columns:
                continue
            b = b.dropna(axis=0)
            b = b.reset_index(drop=True)
            car_names = set(b['Car_RegistrationNumber'].to_list())
            for car in car_names:
                bu = b[b['Car_RegistrationNumber'] == car]
                bu = bu.reset_index(drop=True)
                search(os.path.join(path, filename), args.route_path, args.route_name, bu)

if __name__ == '__main__':
    main()
