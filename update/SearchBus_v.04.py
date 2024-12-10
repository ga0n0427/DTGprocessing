import numpy as np
import pandas as pd
import argparse
import os
import warnings
from datetime import timedelta, datetime

warnings.filterwarnings('ignore')

def calDistance(lat1, lng1, lat2, lng2):
    #두 지리적 좌표 사이의 제곱 거리를 계산합니다.
    return ((lat1 - lat2) ** 2) + ((lng1 - lng2) ** 2)

def parse_time(timestamp):
    #타임스탬프 문자열을 datetime 객체로 변환합니다.
    year = 2000 + int(timestamp[:2])
    month = int(timestamp[2:4])
    day = int(timestamp[4:6])
    hour = int(timestamp[6:8])
    minute = int(timestamp[8:10])
    second = int(timestamp[10:12])
    return datetime(year, month, day, hour, minute, second)

def prepare_directories(bus_path):
    #결과 파일을 저장할 디렉토리를 준비합니다.
    path = bus_path.replace('orderbyNumber', 'sort_file').replace('.csv', '') + '/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def load_bus_route(route_path, route_name):
    #지정된 노선 이름에 해당하는 버스 노선 데이터를 로드합니다.
    bus_route = pd.read_csv(route_path, sep=',', encoding='cp949')
    bus_route = bus_route.astype({'BUS_ROUTE': 'str'})
    return bus_route[bus_route['BUS_ROUTE'] == route_name]

def process_bus_data(bus):
    #버스 데이터를 전처리합니다.
    bus['Information_Occurrence'] = bus['Information_Occurrence'].apply(str)
    bus['Parsed_Date'] = bus['Information_Occurrence'].apply(parse_time)
    bus.sort_values('Parsed_Date', inplace=True)
    bus.reset_index(drop=True, inplace=True)
    return bus

def calculate_mask(bus, bus_route):
    #버스 위치와 가장 가까운 정류소를 찾기 위해 마스크를 계산합니다.
    mask = np.zeros(len(bus), dtype=int)
    bus_lat = bus['LAT'].astype(float).to_numpy()
    bus_lng = bus['LNG'].astype(float).to_numpy()

    for i in range(len(bus)):
        closest_stop_index = np.argmin([calDistance(bus_lat[i], bus_lng[i], float(bus_route.iloc[j]['LAT']), float(bus_route.iloc[j]['LNG'])) for j in range(len(bus_route))]) + 1
        mask[i] = closest_stop_index
    return mask

def filter_unique_mask(mask):
    #마스크에서 유일한 값을 필터링하여 반환합니다.
    m = mask[mask != 0]
    try:
        diff_mask = np.diff(m) != 0
        unique_mask = m[np.insert(diff_mask, 0, True)]
    except IndexError:
        unique_mask = []
    return unique_mask

def save_result(result, path, route_name, bus, num):
    
    #결과 데이터를 저장합니다.
    
    if len(result) > 10:
        result.to_csv(path + route_name + '_' + bus['Car_RegistrationNumber'][0] + '_' + str(num) + '.csv', index=False, encoding='cp949')

def process_mask(mask, bus, bus_route, path, route_name):
    #마스크를 처리하여 결과 데이터를 생성합니다.
    columns = ['BUS_ROUTE', 'STOP_ORD', 'STOP_NAME', 'LAT', 'LNG', 'STOP_ID', 'Information_Occurrence']
    result = pd.DataFrame(columns=columns)
    prev_time = bus['Parsed_Date'].iloc[0]
    num = 0
    before_ord = -1

    for i in range(len(mask)):
        current_time = bus['Parsed_Date'].iloc[i]
        limit_time = prev_time + timedelta(minutes=20)
        prev_time = current_time

        current_ord = int(mask[i] - 1)
        if current_time <= limit_time and before_ord < current_ord:
            current = bus_route.iloc[current_ord]
            current['LAT'] = bus.iloc[i]['LAT']
            current['LNG'] = bus.iloc[i]['LNG']
            current['Information_Occurrence'] = bus.iloc[i]['Information_Occurrence']
            result = pd.concat([result, current.to_frame().T])
            before_ord = current_ord
        else:
            num += 1
            result = pd.DataFrame(columns=columns)
            if current_ord != -1:
                current = bus_route.iloc[current_ord]
                current['LAT'] = bus.iloc[i]['LAT']
                current['LNG'] = bus.iloc[i]['LNG']
                current['Information_Occurrence'] = bus.iloc[i]['Information_Occurrence']
                result = pd.concat([result, current.to_frame().T])
            before_ord = current_ord

    save_result(result, path, route_name, bus, num)

def search(bus_path, route_path, route_name, bus):
   # 주어진 버스 데이터와 노선 데이터를 기반으로 검색을 수행합니다.
 
    path = prepare_directories(bus_path)
    bus_route = load_bus_route(route_path, route_name)
    bus = process_bus_data(bus)
    mask = calculate_mask(bus, bus_route)

    unique_mask = filter_unique_mask(mask)
    print(unique_mask)
    if len(unique_mask) < 10:
        return 0

    if 'H' in bus.columns:
        return

    process_mask(mask, bus, bus_route, path, route_name)

def main():
    #메인 함수로, 명령줄 인수를 처리하고 전체 프로세스를 실행합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('bus_path', help='버스 데이터 폴더 경로')
    parser.add_argument('route_path', help='노선 데이터 파일 경로')
    parser.add_argument('route_name', help='찾고자 하는 노선 이름. -1을 입력하면 모든 노선을 검색합니다.')
    
    args = parser.parse_args()

    for (path, dir, files) in os.walk(args.bus_path):
        for filename in files:
            b = pd.read_csv(os.path.join(path, filename), sep=',', encoding='utf-8')
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
