import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')


# ==== 인덱스 및 초기값 ====
INIT_AVG = 0
MAX_TIME_GAP_MIN = 5  # 5분
NUM_AVERAGE_TYPES = 4

def read_csv_with_fallback(file_path):
    """
    주어진 파일 경로에서 CSV 파일을 utf-8로 먼저 읽고,
    실패하면 cp949로 읽어오는 함수
    
    Parameters:
    file_path (str): 읽을 CSV 파일의 경로
    
    Returns:
    DataFrame: 성공적으로 읽은 데이터프레임
    """
    try:
        # 먼저 utf-8로 읽기 시도
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # utf-8 인코딩 실패 시 cp949로 읽기 시도
        try:
            df = pd.read_csv(file_path, encoding='cp949')
        except Exception as e:
            # cp949로도 실패한 경우 오류 메시지 출력
            print(f"{file_path} 파일을 읽는 데 실패했습니다: {e}")
            return None
    return df


def parse_time(timestamp):
    """
    문자열 또는 Timestamp를 받아 datetime 객체로 변환하는 함수
    """
    if isinstance(timestamp, datetime):
        return timestamp  # 이미 datetime이면 그대로 반환
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)  # 문자열로 변환
    
    timestamp = timestamp.split('.')[0]
    try:
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(timestamp, '%Y-%m-%d %I:%M:%S %p')
        except ValueError:
            return datetime.strptime(timestamp, '%Y-%m-%d %H:%M')

            return datetime.strptime(timestamp, '%Y-%m-%d %H:%M')

def save_busdata(bus_data, file_path):
    bus_data.to_csv(file_path, index = False, encoding = 'cp949')

def devide_up_down(bus_route):
    for i in range(1, len(bus_route)):
        if bus_route.loc[i, 'STOP_ORD'] != bus_route.loc[i - 1, 'STOP_ORD'] + 1:
            return bus_route.loc[i, 'MASK_SELECTED']
    return None

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        
def calculate_average_time(bus_data, route_num, last_stop):
    average_list = [0, 0, 0, 0]
    # MASK_SELECTED 값 1과 2의 시간 차이 계산
    if (not (bus_data[bus_data['MASK_SELECTED'] == 1].empty)) and (not bus_data[bus_data['MASK_SELECTED'] == 2].empty):
        time_difference = bus_data[bus_data['MASK_SELECTED'] == 2]['Parsed_Date'].values[0] - bus_data[bus_data['MASK_SELECTED'] == 1]['Parsed_Date'].values[0]
        average_list[0] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    # MASK_SELECTED 값 27과 28의 시간 차이 계산
    if (not bus_data[bus_data['MASK_SELECTED'] == (route_num-2)].empty) and (not bus_data[bus_data['MASK_SELECTED'] == (route_num-1)].empty):
        time_difference = bus_data[bus_data['MASK_SELECTED'] == (route_num-1)]['Parsed_Date'].values[0] - bus_data[bus_data['MASK_SELECTED'] == route_num-2]['Parsed_Date'].values[0] 
        average_list[1] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    # MASK_SELECTED 값 29와 30의 시간 차이 계산
    if (not bus_data[bus_data['MASK_SELECTED'] == route_num].empty) and (not bus_data[bus_data['MASK_SELECTED'] == (route_num+1)].empty):
        time_difference = bus_data[bus_data['MASK_SELECTED'] == (route_num + 1)]['Parsed_Date'].values[0] - bus_data[bus_data['MASK_SELECTED'] == route_num]['Parsed_Date'].values[0] 
        average_list[2] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    # MASK_SELECTED 값 56과 57의 시간 차이 계산
    if (not bus_data[bus_data['MASK_SELECTED'] == (last_stop-1)].empty) and (not bus_data[bus_data['MASK_SELECTED'] == last_stop].empty):
        time_difference = bus_data[bus_data['MASK_SELECTED'] == last_stop]['Parsed_Date'].values[0] -  bus_data[bus_data['MASK_SELECTED'] == (last_stop-1)]['Parsed_Date'].values[0]
        average_list[3] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    return average_list
    
def filter_function(bus_data, route_num, last_stop, direction):
    """
    상행/하행 여부에 따라 검증 조건을 다르게 적용하는 함수

    Parameters:
    - bus_data: 처리할 데이터프레임
    - route_num: 상하행 기준 번호
    - last_stop: 전체 정류장 수
    - direction: 'up' 또는 'down'

    Returns:
    - (DataFrame or None, int): 조건 통과한 데이터와 통과 코드
    """
    bus_data['MASK_SELECTED'] = bus_data['MASK_SELECTED'].astype(int)

    if direction == 'up':
        if 1 not in bus_data['MASK_SELECTED'].values and 2 not in bus_data['MASK_SELECTED'].values:
            return None, 2
        if (route_num - 1) not in bus_data['MASK_SELECTED'].values and (route_num - 2) not in bus_data['MASK_SELECTED'].values:
            return None, 2
        if 1 not in bus_data['MASK_SELECTED'].values:
            return None, 1
        if (route_num - 1) not in bus_data['MASK_SELECTED'].values:
            return None, 1

    elif direction == 'down':
        if route_num not in bus_data['MASK_SELECTED'].values and (route_num + 1) not in bus_data['MASK_SELECTED'].values:
            return None, 2
        if last_stop not in bus_data['MASK_SELECTED'].values and (last_stop - 1) not in bus_data['MASK_SELECTED'].values:
            return None, 2
        if route_num not in bus_data['MASK_SELECTED'].values:
            return None, 1
        if last_stop not in bus_data['MASK_SELECTED'].values:
            return None, 1

    return bus_data, 1



#data 처리 함수 
def process_data(file_path, bus_stop_data, route_num, last_stop, direction):
    bus_data = read_csv_with_fallback(file_path)
    bus_data['Parsed_Date'] = bus_data['Information_Occurrence'].apply(parse_time)

    # 🔽 수정된 filter_function 호출
    bus_data, check_num = filter_function(bus_data, route_num, last_stop, direction)

    if bus_data is None:
        return None, [0, 0, 0, 0], check_num
    else:
        average_list = calculate_average_time(bus_data, route_num, last_stop)
        num_data = bus_data['MASK_SELECTED'].values
        all_stations = np.arange(1, len(bus_stop_data))
        missing_stations = np.setdiff1d(all_stations, num_data)
        missing_stations = group_numbers(missing_stations)
        bus_data = fill_space(bus_data, bus_stop_data, missing_stations)
        return bus_data, average_list, check_num
    
#비는 시간 계산해서 채우는 함수 
def fill_space(bus_data, bus_stop_data, missing_stations):
    for group in missing_stations:
        first_missing = group[0]
        last_missing = group[-1]
        
        previous_value = first_missing - 1
        # 이후 값 찾기
        next_value = last_missing + 1 
        # previous_value 또는 next_value와 일치하는 'On' 열 값만 선택
        slice_bus_df = bus_data[(bus_data['MASK_SELECTED'] == previous_value) | (bus_data['MASK_SELECTED'] == next_value)]
        
        slice_bus_stop_df = bus_stop_data[(bus_stop_data['MASK_SELECTED'] >= previous_value) & (bus_stop_data['MASK_SELECTED'] < next_value)]
        calculate_df = calculate_time(slice_bus_df, slice_bus_stop_df, group)
        before_df = bus_data[bus_data['MASK_SELECTED'] <= previous_value]
        after_df = bus_data[bus_data['MASK_SELECTED'] > previous_value]
        bus_data = pd.concat([before_df, calculate_df, after_df]).reset_index(drop=True)
        bus_data = bus_data[['MASK_SELECTED', 'Information_Occurrence', 'BUS_ROUTE', 'STOP_ORD', 'STOP_NAME', 'LAT', 'LNG', 'STOP_ID']]
    return bus_data

def calculate_time(slice_bus_df, slice_bus_stop_df, group):
    slice_bus_df.loc[:, 'Parsed_Date'] = slice_bus_df['Information_Occurrence'].apply(parse_time)
    time_difference = slice_bus_df.iloc[-1]['Parsed_Date'] - slice_bus_df.iloc[0]['Parsed_Date']
    total_time_seconds = time_difference.total_seconds()
    total_distance = slice_bus_stop_df['next'].sum()
    
    # 방어: 거리가 0이면 division 에러 방지
    if total_distance == 0:
        return pd.DataFrame(columns=slice_bus_stop_df.columns)

    speed_per_meter = total_time_seconds / total_distance
    filled_times = []
    accumulated_time = 0

    for i in range(len(slice_bus_stop_df) - 1):
        segment_distance = slice_bus_stop_df.iloc[i]['next']
        if slice_bus_stop_df.iloc[i + 1]['MASK_SELECTED'] in group:
            segment_time_seconds = segment_distance * speed_per_meter
            accumulated_time += segment_time_seconds
            base_datetime = slice_bus_df.iloc[0]['Parsed_Date']
            filled_time_datetime = (base_datetime + timedelta(seconds=accumulated_time)).replace(microsecond=0)

            filled_times.append({
                'MASK_SELECTED': slice_bus_stop_df.iloc[i + 1]['MASK_SELECTED'],
                'Information_Occurrence': filled_time_datetime
            })

    # 방어 코드 추가: 데이터가 없을 경우 바로 빈 DataFrame 반환
    if not filled_times:
        return pd.DataFrame(columns=slice_bus_stop_df.columns)

    filled_times_df = pd.DataFrame(filled_times)
    combined_df = pd.merge(filled_times_df, slice_bus_stop_df, on='MASK_SELECTED', how='left')
    combined_df = combined_df[['MASK_SELECTED', 'Information_Occurrence', 'BUS_ROUTE', 'STOP_ORD', 'STOP_NAME', 'LAT', 'LNG', 'STOP_ID']]
    return combined_df

#비는 번호들 찾아서 연속이면 연속으로 묶어주는 함수         
def group_numbers(missing_stations):
    if len(missing_stations) == 0:
        return []
    
    grouped = []
    group = [missing_stations[0]]  # 첫 번째 요소로 그룹 시작
    
    for i in range(1, len(missing_stations)):
        # 현재 값과 이전 값의 차이가 1이면 연속된 숫자
        if missing_stations[i] == missing_stations[i-1] + 1:
            group.append(missing_stations[i])
        else:
            # 연속되지 않으면 현재 그룹을 추가하고 새로운 그룹 시작
            grouped.append(group)
            group = [missing_stations[i]]
    
    # 마지막 그룹도 추가
    grouped.append(group)
    return grouped
    
def fill_space_average(file_path, total_average_list, bus_stop_data, route_num, last_stop):
    bus_data = read_csv_with_fallback(file_path)
    bus_data['Parsed_Date'] = bus_data['Information_Occurrence'].apply(parse_time)

    # ✅ MASK_SELECTED 1이 없고 2는 있을 경우
    if 1 not in bus_data['MASK_SELECTED'].values and 2 in bus_data['MASK_SELECTED'].values:
        second_rows = bus_data.loc[bus_data['MASK_SELECTED'] == 2, 'Parsed_Date']
        if not second_rows.empty:
            second_parsed_date = second_rows.iloc[0]
            new_date = (second_parsed_date - pd.to_timedelta(total_average_list[0], unit='s')).replace(microsecond=0)
            stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == 1].iloc[0].to_dict()
            stop_info.update({'MASK_SELECTED': 1, 'Information_Occurrence': new_date})
            new_row = stop_info
            bus_data = pd.concat([pd.DataFrame([new_row]), bus_data], ignore_index=True)
            bus_data = bus_data.sort_values(by='MASK_SELECTED').reset_index(drop=True)

    # ✅ MASK_SELECTED 28이 없고 27은 있을 경우
    if (route_num - 1) not in bus_data['MASK_SELECTED'].values and (route_num - 2) in bus_data['MASK_SELECTED'].values:
        twoseven_rows = bus_data.loc[bus_data['MASK_SELECTED'] == (route_num - 2), 'Parsed_Date']
        if not twoseven_rows.empty:
            parsed_date = twoseven_rows.iloc[0]
            new_date = (parsed_date + pd.to_timedelta(total_average_list[1], unit='s')).replace(microsecond=0)
            stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == (route_num - 1)].iloc[0].to_dict()
            stop_info.update({'MASK_SELECTED': (route_num - 1), 'Information_Occurrence': new_date})
            new_row = stop_info
            idx = bus_data[bus_data['MASK_SELECTED'] == (route_num - 2)].index[0] + 1
            bus_data = pd.concat([bus_data.iloc[:idx], pd.DataFrame([new_row]), bus_data.iloc[idx:]], ignore_index=True)

    # ✅ MASK_SELECTED 29이 없고 30은 있을 경우
    if route_num not in bus_data['MASK_SELECTED'].values and (route_num + 1) in bus_data['MASK_SELECTED'].values:
        thirty_rows = bus_data.loc[bus_data['MASK_SELECTED'] == (route_num + 1), 'Parsed_Date']
        if not thirty_rows.empty:
            parsed_date = thirty_rows.iloc[0]
            new_date = (parsed_date - pd.to_timedelta(total_average_list[2], unit='s')).replace(microsecond=0)
            stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == route_num].iloc[0].to_dict()
            stop_info.update({'MASK_SELECTED': route_num, 'Information_Occurrence': new_date})
            new_row = stop_info
            idx = bus_data[bus_data['MASK_SELECTED'] == (route_num + 1)].index[0]
            bus_data = pd.concat([bus_data.iloc[:idx], pd.DataFrame([new_row]), bus_data.iloc[idx:]], ignore_index=True)

    # ✅ MASK_SELECTED 57이 없고 56은 있을 경우
    if last_stop not in bus_data['MASK_SELECTED'].values and (last_stop - 1) in bus_data['MASK_SELECTED'].values:
        fsix_rows = bus_data.loc[bus_data['MASK_SELECTED'] == (last_stop - 1), 'Parsed_Date']
        if not fsix_rows.empty:
            parsed_date = fsix_rows.iloc[0]
            new_date = (parsed_date + pd.to_timedelta(total_average_list[3], unit='s')).replace(microsecond=0)
            stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == last_stop].iloc[0].to_dict()
            stop_info.update({'MASK_SELECTED': last_stop, 'Information_Occurrence': new_date})
            new_row = stop_info
            bus_data = pd.concat([bus_data, pd.DataFrame([new_row])], ignore_index=True)

    return bus_data



def main():
    """
    버스 정류장 데이터 결측치 채우기 (상/하행 분리)
    """

    parser = argparse.ArgumentParser(description='버스 정류장 데이터 결측치 채우기')
    parser.add_argument('--bus_path', type=str, help='처리할 데이터 폴더 경로')
    parser.add_argument('--route_path', type=str, help='처리할 버스 정류장 데이터 경로 (CSV 파일)')
    args = parser.parse_args()
    
    # 경로 설정
    bus_path = args.bus_path
    route_path = args.route_path

    # 정류장 정보 및 경계값
    bus_stop_data = read_csv_with_fallback(route_path)
    route_num = devide_up_down(bus_stop_data)
    last_stop = len(bus_stop_data)

    # 상/하행 분리
    up_stop_data = bus_stop_data[bus_stop_data['MASK_SELECTED'] < route_num].reset_index(drop=True)
    down_stop_data = bus_stop_data[bus_stop_data['MASK_SELECTED'] >= route_num].reset_index(drop=True)

    # 평균 시간 누적 리스트
    total_average_up = [0, 0, 0, 0]
    total_count_up = [0, 0, 0, 0]
    total_average_down = [0, 0, 0, 0]
    total_count_down = [0, 0, 0, 0]
    need_fill_list = []

    for root, dirs, files in os.walk(bus_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                # ✅ 파일명에서 상/하행 판단
                if 'up' in file:
                    direction = 'up'
                    stop_data = up_stop_data
                elif 'down' in file:
                    direction = 'down'
                    stop_data = down_stop_data
                else:
                    continue

                bus_data, average_list, check_num = process_data(file_path, stop_data, route_num, last_stop, direction)

                if check_num == 1:
                    need_fill_list.append((file_path, direction))
                elif check_num == 2:
                    delete_file(file_path)
                    continue

                if bus_data is not None:
                    save_busdata(bus_data, file_path)

                for i in range(4):
                    if average_list[i] != 0:
                        if direction == 'up':
                            total_count_up[i] += 1
                            total_average_up[i] += average_list[i]
                        else:
                            total_count_down[i] += 1
                            total_average_down[i] += average_list[i]

    average_result_up = [int(total_average_up[i] / total_count_up[i]) if total_count_up[i] > 0 else 0 for i in range(4)]
    average_result_down = [int(total_average_down[i] / total_count_down[i]) if total_count_down[i] > 0 else 0 for i in range(4)]

    print("평균 상행:", average_result_up)
    print("평균 하행:", average_result_down)

    for file_path, direction in need_fill_list:
        stop_data = up_stop_data if direction == 'up' else down_stop_data
        avg = average_result_up if direction == 'up' else average_result_down

        bus_data = fill_space_average(file_path, avg, stop_data, route_num, last_stop)
        num_data = bus_data['MASK_SELECTED'].values
        all_stations = np.arange(1, len(stop_data))
        missing_stations = np.setdiff1d(all_stations, num_data)
        missing_stations = group_numbers(missing_stations)
        bus_data = fill_space(bus_data, stop_data, missing_stations)
        save_busdata(bus_data, file_path)



if __name__ == '__main__':
    main()
