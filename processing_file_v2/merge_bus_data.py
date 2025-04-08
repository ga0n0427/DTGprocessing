import os
import pandas as pd
import argparse
from datetime import datetime
import re

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


def merge_files(input_folder, output_file, last_num):
    """
    주어진 폴더 내의 모든 CSV 파일을 하나로 합치는 함수
    
    Parameters:
    input_folder (str): CSV 파일들이 위치한 폴더 경로
    output_file (str): 합쳐진 결과를 저장할 파일 경로
    last_num (int): MASKED_SELECTED 열의 마지막 번호
    """
    merged_df = None
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                match = re.search(r'\d+[가-힣]+(\d+)', file).group(1)
                file_path = os.path.join(root, file)
                df = read_csv_with_fallback(file_path)
                check_result = check_masked_selected(df, last_num)
                if check_result == None:
                    print(f"{file_path} 파일이 검증에 실패하여 건너뜁니다.")
                    continue
                df['Bus_num'] = match
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
    if merged_df is not None:
        # Information_Occurrence를 datetime 객체로 변환하여 Parsed_Date 열 생성
        merged_df['Parsed_Date'] = merged_df['Information_Occurrence'].apply(parse_time)

        # MASK_SELECTED가 1인 경우를 기준으로 노선을 그룹화 (Route_Group 생성)
        merged_df['Route_Group'] = (merged_df['MASK_SELECTED'] == 1).cumsum()

        # MASK_SELECTED가 1인 행을 기준으로 각 노선의 첫 번째 정류장을 찾음
        first_stops = merged_df[merged_df['MASK_SELECTED'] == 1]

        # 각 노선의 첫 번째 정류장을 시간 기준으로 정렬
        first_stops_sorted = first_stops.sort_values(by='Parsed_Date')

        # 중복된 첫 번째 정류장(같은 Bus_num, Parsed_Date)을 찾음
        duplicate_groups = first_stops_sorted[first_stops_sorted.duplicated(subset=['Bus_num', 'Parsed_Date'], keep=False)]['Route_Group'].unique()

        # 중복된 데이터에서 하나만 남기고 나머지 삭제
        first_stops_dedup = first_stops_sorted.drop_duplicates(subset=['Bus_num', 'Parsed_Date'], keep='first')

        # 중복되지 않은 첫 번째 정류장을 기준으로 전체 데이터를 재정렬
        sorted_df = pd.DataFrame()

        for _, first_stop in first_stops_dedup.iterrows():
            route_group = merged_df[merged_df['Route_Group'] == first_stop['Route_Group']]
            sorted_df = pd.concat([sorted_df, route_group], ignore_index=True)

        # 더 이상 필요하지 않다면 Parsed_Date 및 Route_Group 열 삭제
        sorted_df = sorted_df.drop(['Parsed_Date', 'Route_Group'], axis=1)

        # 결과를 CSV로 저장
        sorted_df.to_csv(output_file, index=False)
        print(f"모든 파일이 합쳐져 {output_file}에 저장되었습니다.")
    else:
        print("합칠 파일이 없습니다.")
def check_masked_selected(df, last_num):
    """
    MASKED_SELECTED 열에 대한 세 가지 검증 기능을 수행하는 함수
    
    Parameters:
    df (DataFrame): 확인할 데이터프레임
    last_num (int): MASK_SELECTEDD 열의 마지막 번호
    
    Returns:
    int: 모든 검증이 통과되면 0, 하나라도 실패하면 None
    """
    # 첫 번째 검증: 동일한 값이 두 개 이상 있는 경우
    if df.duplicated(subset=['MASK_SELECTED'], keep=False).any():
        return None
    
    # 두 번째 검증: 1에서 last_num까지 빈 번호가 있는 경우
    masked_values = df['MASK_SELECTED'].dropna().astype(int).unique()
    missing_numbers = [i for i in range(1, last_num + 1) if i not in masked_values]
    if missing_numbers:
        return None
    
    # 세 번째 검증: 순서대로 되어 있지 않은 경우
    if (df['MASK_SELECTED'].dropna().astype(int).diff() < 0).any():
        return None
    
    return 0



def main():
    # argparse를 사용하여 bus_path와 route_path를 입력받습니다.
    parser = argparse.ArgumentParser(description="Process bus and route data")
    parser.add_argument('--bus_path', type=str, help="Path to the bus data folder")
    parser.add_argument('--route_path', type=str, help="Path to the route info CSV file")
    parser.add_argument('--output_path', type=str, help="Path to the route info CSV file")
    
    args = parser.parse_args()
    route_path = args.route_path
    bus_route = read_csv_with_fallback(route_path)
    
    last_num = len(bus_route)
    merge_files(args.bus_path, args.output_path, last_num)
    
if __name__ == "__main__":
    main()