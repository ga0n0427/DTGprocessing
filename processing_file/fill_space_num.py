import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')

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
    'YYYY-MM-DD HH:MM:SS AM/PM', 'YYYY-MM-DD HH:MM:SS', 또는 'YYYY-MM-DD HH:MM' 형식을 파싱하는 함수
    """
    timestamp = timestamp.split('.')[0]
    try:
        # 먼저 24시간 형식으로 파싱 시도
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # 실패하면 12시간 형식으로 파싱 시도
            return datetime.strptime(timestamp, '%Y-%m-%d %I:%M:%S %p')
        except ValueError:
            # 실패하면 'YYYY-MM-DD HH:MM' 형식으로 파싱 시도
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
        time_difference = bus_data[bus_data['MASK_SELECTED'] == 28]['Parsed_Date'].values[0] - bus_data[bus_data['MASK_SELECTED'] == 27]['Parsed_Date'].values[0] 
        average_list[1] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    # MASK_SELECTED 값 29와 30의 시간 차이 계산
    if (not bus_data[bus_data['MASK_SELECTED'] == route_num].empty) and (not bus_data[bus_data['MASK_SELECTED'] == (route_num+1)].empty):
        time_difference = bus_data[bus_data['MASK_SELECTED'] == 30]['Parsed_Date'].values[0] - bus_data[bus_data['MASK_SELECTED'] == 29]['Parsed_Date'].values[0] 
        average_list[2] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    # MASK_SELECTED 값 56과 57의 시간 차이 계산
    if (not bus_data[bus_data['MASK_SELECTED'] == (last_stop-1)].empty) and (not bus_data[bus_data['MASK_SELECTED'] == last_stop].empty):
        time_difference = bus_data[bus_data['MASK_SELECTED'] == 57]['Parsed_Date'].values[0] -  bus_data[bus_data['MASK_SELECTED'] == 56]['Parsed_Date'].values[0]
        average_list[3] = time_difference / np.timedelta64(1, 's')  # 초 단위로 변환

    return average_list
    
def filter_function(bus_data, route_num, last_stop):
    # 데이터 타입을 맞추기 위해 MASK_SELECTED를 정수형으로 변환
    bus_data['MASK_SELECTED'] = bus_data['MASK_SELECTED'].astype(int)

    # 조건 1: MASK_SELECTED == 1, 2가 모두 없으면 None, 2 리턴
    if 1 not in bus_data['MASK_SELECTED'].values and 2 not in bus_data['MASK_SELECTED'].values:
        print(1)
        return None, 2

    # 조건 2: route_num-1, route_num-2가 모두 없으면 None, 2 리턴
    if (route_num - 1) not in bus_data['MASK_SELECTED'].values and (route_num - 2) not in bus_data['MASK_SELECTED'].values:
        print(1)
        return None, 2

    # 조건 3: route_num, route_num+1이 모두 없으면 None, 2 리턴
    if route_num not in bus_data['MASK_SELECTED'].values and (route_num + 1) not in bus_data['MASK_SELECTED'].values:
        print(1)
        return None, 2

    # 조건 4: last_stop, last_stop-1이 모두 없으면 None, 2 리턴
    if last_stop not in bus_data['MASK_SELECTED'].values and (last_stop - 1) not in bus_data['MASK_SELECTED'].values:
        print(1)
        return None, 2

    # 조건 5: 1이 없으면 None, 1 리턴
    if 1 not in bus_data['MASK_SELECTED'].values:
        return None, 1

    # 조건 6: route_num-1이 없으면 None, 1 리턴
    if (route_num - 1) not in bus_data['MASK_SELECTED'].values:
        return None, 1

    # 조건 7: route_num이 없으면 None, 1 리턴
    if route_num not in bus_data['MASK_SELECTED'].values:
        return None, 1

    # 조건 8: last_stop이 없으면 None, 1 리턴
    if last_stop not in bus_data['MASK_SELECTED'].values:
        return None, 1

    # 모든 조건에 해당하지 않으면 bus_data와 1을 리턴
    
    return bus_data, 1


#data 처리 함수 
def process_data(file_path, bus_stop_data, route_num, last_stop):
    bus_data = read_csv_with_fallback(file_path)
    # parse_time 함수 적용하여 시간과 날짜 분리
    bus_data['Parsed_Date'] = bus_data['Information_Occurrence'].apply(parse_time)
    bus_data, check_num = filter_function(bus_data, route_num, last_stop)
    if bus_data is None:
        return None, [0,0,0,0], check_num
    else:
        average_list = calculate_average_time(bus_data, route_num, last_stop)
        num_data = bus_data['MASK_SELECTED'].values
        all_stations = np.arange(1, len(bus_stop_data))
        missing_stations =  np.setdiff1d(all_stations, num_data)  #비는 정류장 찾기
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
    # 이전 시간과 다음 시간 차이 계산 (1번과 4번 정류장 시간 차이)
    time_difference = slice_bus_df.iloc[-1]['Parsed_Date'] - slice_bus_df.iloc[0]['Parsed_Date']
    
    total_time_seconds = time_difference.total_seconds()  # 시간 차이를 초 단위로 변환
    # 1번부터 3번 정류장까지의 총 거리를 계산 (next 열의 총합)
    total_distance = slice_bus_stop_df['next'].sum()
    # 1미터당 걸리는 시간(초) 계산
    speed_per_meter = total_time_seconds / total_distance
    # filled_times를 데이터프레임으로 만들기 위해 리스트 생성
    filled_times = []
    accumulated_time = 0
    
    # 결측치가 있는 2번과 3번 구간에 대한 시간 계산
    for i in range(len(slice_bus_stop_df) - 1):
        segment_distance = slice_bus_stop_df.iloc[i]['next']  # 현재 구간의 거리
        
        # 2번과 3번 구간에 해당하는 거리 범위만 처리
        if slice_bus_stop_df.iloc[i + 1]['MASK_SELECTED'] in group:
            segment_time_seconds = segment_distance * speed_per_meter  # 해당 구간에 걸리는 시간 (초)
            accumulated_time += segment_time_seconds
            base_datetime = slice_bus_df.iloc[0]['Parsed_Date']
            filled_time_datetime = (base_datetime + timedelta(seconds=accumulated_time)).replace(microsecond=0)
            
            # MASK_SELECTED와 시간을 함께 저장
            filled_times.append({'MASK_SELECTED': slice_bus_stop_df.iloc[i + 1]['MASK_SELECTED'], 'Information_Occurrence': filled_time_datetime})

    
    # filled_times 리스트를 데이터프레임으로 변환
    filled_times_df = pd.DataFrame(filled_times)
    
    # slice_bus_stop_df에 있는 관련 정보를 filled_times_df와 결합
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
    
    # Add missing MASK_SELECTED 1 if it does not exist
    if 1 not in bus_data['MASK_SELECTED'].values:
        print(file_path)
        second_parsed_date = bus_data.loc[bus_data['MASK_SELECTED'] == 2, 'Parsed_Date'].iloc[0]
        new_date = (second_parsed_date - pd.to_timedelta(total_average_list[0], unit='s')).replace(microsecond=0)
        
        # Get additional information from bus_stop_data and create new row
        stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == 1].iloc[0].to_dict()
        stop_info.update({'MASK_SELECTED': 1, 'Information_Occurrence': new_date})
        new_row = stop_info
        
        bus_data = pd.concat([pd.DataFrame([new_row]), bus_data], ignore_index=True)
        bus_data = bus_data.sort_values(by='MASK_SELECTED').reset_index(drop=True)
    
    # Add missing MASK_SELECTED 28 if it does not exist
    if (route_num - 1) not in bus_data['MASK_SELECTED'].values:
        twenty_seventh_parsed_date = bus_data.loc[bus_data['MASK_SELECTED'] == (route_num - 2), 'Parsed_Date'].iloc[0]
        new_date = (twenty_seventh_parsed_date + pd.to_timedelta(total_average_list[1], unit='s')).replace(microsecond=0)
        
        stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == (route_num - 1)].iloc[0].to_dict()
        stop_info.update({'MASK_SELECTED': (route_num - 1), 'Information_Occurrence': new_date})
        new_row = stop_info
        
        idx = bus_data[bus_data['MASK_SELECTED'] == (route_num - 2)].index[0] + 1
        bus_data = pd.concat([bus_data.iloc[:idx], pd.DataFrame([new_row]), bus_data.iloc[idx:]], ignore_index=True)
    
    # Add missing MASK_SELECTED 29 if it does not exist
    if route_num not in bus_data['MASK_SELECTED'].values:
        thirtieth_parsed_date = bus_data.loc[bus_data['MASK_SELECTED'] == (route_num + 1), 'Parsed_Date'].iloc[0]
        new_date = (thirtieth_parsed_date - pd.to_timedelta(total_average_list[2], unit='s')).replace(microsecond=0)
        
        stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == route_num].iloc[0].to_dict()
        stop_info.update({'MASK_SELECTED': route_num, 'Information_Occurrence': new_date})
        new_row = stop_info
        
        idx = bus_data[bus_data['MASK_SELECTED'] == (route_num + 1)].index[0]
        bus_data = pd.concat([bus_data.iloc[:idx], pd.DataFrame([new_row]), bus_data.iloc[idx:]], ignore_index=True)
    
    # Add missing MASK_SELECTED 57 if it does not exist
    if last_stop not in bus_data['MASK_SELECTED'].values:
        fifty_sixth_parsed_date = bus_data.loc[bus_data['MASK_SELECTED'] == (last_stop - 1), 'Parsed_Date'].iloc[0]
        new_date = (fifty_sixth_parsed_date + pd.to_timedelta(total_average_list[3], unit='s')).replace(microsecond=0)
        
        stop_info = bus_stop_data.loc[bus_stop_data['MASK_SELECTED'] == last_stop].iloc[0].to_dict()
        stop_info.update({'MASK_SELECTED': last_stop, 'Information_Occurrence': new_date})
        new_row = stop_info
        
        bus_data = pd.concat([bus_data, pd.DataFrame([new_row])], ignore_index=True)
    return bus_data


def main():
    parser = argparse.ArgumentParser(description='버스 정류장 데이터 결측치 채우기')
    parser.add_argument('--bus_path', type=str, help='처리할 데이터 폴더 경로')
    parser.add_argument('--route_path', type=str, help='처리할 버스 정류장 데이터 경로 (CSV 파일)')
    args = parser.parse_args()
    
    
    bus_stop_data = read_csv_with_fallback(args.route_path)
    total_average_list = [0,0,0,0]
    total_count_list = [0,0,0,0]
    need_fill_list = []
    route_num = devide_up_down(bus_stop_data)
    last_stop = len(bus_stop_data)
    
    for root, dirs, files in os.walk(args.bus_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                bus_data, average_list, check_num = process_data(file_path, bus_stop_data, route_num, last_stop)
                if check_num == 1:
                    need_fill_list.append(file_path)
                elif check_num == 2:
                    delete_file(file_path)
                if bus_data is None:
                    pass
                else:
                    save_busdata(bus_data, file_path)
                for i in range(len(average_list)):
                    if average_list[i] != 0:
                        total_count_list[i] += 1
                        total_average_list[i] += average_list[i]
    print(total_average_list)
    average_result = [0,0,0,0]                  
    for i in range(len(average_result)):
        average_result[i] = int(total_average_list[i] / total_count_list[i])
    print(average_result)
    for fill in need_fill_list:
        bus_data = fill_space_average(fill, average_result, bus_stop_data, route_num, last_stop)
        num_data = bus_data['MASK_SELECTED'].values
        all_stations = np.arange(1, len(bus_stop_data))
        missing_stations =  np.setdiff1d(all_stations, num_data)  #비는 정류장 찾기
        missing_stations = group_numbers(missing_stations)
        bus_data = fill_space(bus_data, bus_stop_data, missing_stations)
        save_busdata(bus_data, fill)

if __name__ == '__main__':
    main()
