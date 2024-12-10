import os
import pandas as pd
from datetime import datetime

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

def process_csv_file(file_path, save_base_path, base_path, bus_stop_count, file_counter):
    # CSV 파일을 읽습니다.
    df = read_csv_with_fallback(file_path)
    
    # Information_Occurrence 열 파싱
    df = parse_information_occurrence(df)
    
    # 파일의 행 수를 계산하여, 버스 정류장 수 + (버스 정류장 수 / 4) 계산
    max_allowed_rows = bus_stop_count + (bus_stop_count // 4)
    
    if len(df) > max_allowed_rows:
        # 파일의 행 수가 기준을 넘는 경우
        df = handle_large_file(df, file_path, save_base_path, base_path, file_counter)
    else:
        # 현재 행이 이전 행보다 작은 값이 나왔을 때 그 행을 삭제하는 함수 호출
        df = handle_decreasing_rows(df)
    
    # 연속된 중복 값이 있을 경우 처리하는 함수 호출
    df = remove_consecutive_duplicates(df, 'MASK_SELECTED')
    
    # 날짜에 맞는 폴더 내의 파일 개수를 기반으로 파일 순서를 추적
    date_folder_path = create_date_folder_path(save_base_path, df['Date'].iloc[0])
    
    # 폴더가 없으면 생성합니다.
    os.makedirs(date_folder_path, exist_ok=True)
    
    file_count_in_date_folder = len([f for f in os.listdir(date_folder_path) if os.path.isfile(os.path.join(date_folder_path, f))])

    # 파일 저장 경로 계산 (월, 일에 맞는 폴더로 저장, 파일 개수에 따른 숫자 추가)
    save_file_path = create_save_path(file_path, save_base_path, base_path, df['Date'].iloc[0], file_count_in_date_folder + 1)
    
    # 파일 저장 경로가 존재하지 않으면 폴더를 생성
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    
    # 처리된 데이터를 저장 (Date와 Time 열을 제외하고 저장)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.to_csv(save_file_path, index=False)
    print(f"Saved processed file to: {save_file_path}")

def remove_consecutive_duplicates(df, column_name):
    """
    연속된 중복 값이 있는 경우 첫 번째 값만 남기고 나머지는 삭제하는 함수
    """
    df = df.loc[df[column_name].shift() != df[column_name]]
    return df

def handle_decreasing_rows(df):
    """
    현재 행의 값이 이전 행의 값보다 작을 때 해당 행을 삭제하는 함수.
    """
    i = 1
    while i < len(df):
        current_value = df.loc[i, 'MASK_SELECTED']
        previous_value = df.loc[i - 1, 'MASK_SELECTED']
        
        # 현재 값이 이전 값보다 작으면 해당 행을 삭제
        if current_value < previous_value:
            df = df.drop(i).reset_index(drop=True)
        else:
            i += 1  # 정상적인 경우 다음 행으로 이동

    return df


def handle_large_file(df, file_path, save_base_path, base_path, file_counter):
    """
    행 수가 기준을 넘는 경우 처리하는 함수. 큰 값에서 작은 값으로 변화하는 부분을 찾아
    큰 값까지를 새로운 파일로 저장하고 이후 데이터를 삭제한 후 전처리합니다.
    """
    # 먼저 clean_mask_selected를 사용하여 한 번 전처리 진행
    df = clean_mask_selected(df)

    # 전처리 후 큰 값에서 작은 값으로 변화하는 부분을 탐지
    for i in range(1, len(df)):
        current_value = df.loc[i, 'MASK_SELECTED']
        previous_value = df.loc[i - 1, 'MASK_SELECTED']
        
        # 큰 값에서 작은 값으로 변화하는 경우 (예: 27에서 2로 작아짐)
        if previous_value > 20 and current_value < previous_value:
            # 이전 값까지의 데이터(큰 값까지)를 저장할 파일 생성
            date_folder_path = create_date_folder_path(save_base_path, df['Date'].iloc[0])
            file_count_in_date_folder = len([f for f in os.listdir(date_folder_path) if os.path.isfile(os.path.join(date_folder_path, f))])
            
            save_file_path = create_save_path(file_path, save_base_path, base_path, df['Date'].iloc[0], file_count_in_date_folder + 1)
            df_part = df.iloc[:i]  # 큰 값까지의 데이터
            
            # 새로운 파일로 저장 (Date와 Time 열을 제외하고 저장)
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            df_part.drop(columns=['Date', 'Time'], inplace=True)
            df_part.to_csv(save_file_path, index=False)
            print(f"Saved part of the file to: {save_file_path}")
            
            # 그 이후 데이터로 전처리 진행 (큰 값 이후 데이터 삭제)
            df = df.iloc[i:].reset_index(drop=True)
            break

    return df

def clean_mask_selected(df):
    """
    MASK_SELECTED 열을 기준으로 비정상적으로 값이 변화한 행을 삭제하는 함수.
    값이 이전 값보다 3 이상 차이가 나거나 20분 이상의 시간 차이가 발생한 경우, 해당 행을 삭제.
    """
    i = 1
    while i < len(df):
        if i > 0:  # 첫 번째 행은 이전 값을 참조할 수 없으므로 건너뜁니다.
            current_value = df.loc[i, 'MASK_SELECTED']
            previous_value = df.loc[i - 1, 'MASK_SELECTED']
            current_time = datetime.strptime(df.loc[i, 'Information_Occurrence'], '%Y-%m-%d %H:%M:%S')
            previous_time = datetime.strptime(df.loc[i - 1, 'Information_Occurrence'], '%Y-%m-%d %H:%M:%S')

            # 이전 값과 현재 값의 차이가 3 초과이거나, 시간 차이가 20분 초과인 경우
            if (current_value < previous_value or abs(current_value - previous_value) > 3) or ((current_time - previous_time).total_seconds() > 1200):
                df = df.drop(i - 1).reset_index(drop=True)
                i -= 1  # 행이 삭제되었으므로 인덱스 조정
            else:
                i += 1  # 정상적인 경우 다음 행으로 이동
        else:
            i += 1  # i = 0일 경우 그냥 넘어갑니다.

    return df


def parse_information_occurrence(df):
    """
    Information_Occurrence 열을 날짜(YYYY-MM-DD)와 시간(HH:MM:SS)로 분리하는 함수
    """
    def parse_datetime(entry):
        # '2020-06-23 06:37:03' 형식으로 파싱
        return datetime.strptime(entry, '%Y-%m-%d %H:%M:%S')

    # Information_Occurrence 열에서 날짜와 시간을 분리
    df['Date'] = df['Information_Occurrence'].apply(lambda x: parse_datetime(x).date())
    df['Time'] = df['Information_Occurrence'].apply(lambda x: parse_datetime(x).time())
    
    return df

def create_date_folder_path(save_base_path, date):
    """
    월, 일에 맞는 폴더 경로를 생성하는 함수
    """
    month = date.strftime("%B")  # 예: June
    day = f"day{date.day}"  # 예: day21
    return os.path.join(save_base_path, month, day)

def create_save_path(file_path, save_base_path, base_path, date, file_counter):
    """
    원본 파일 경로에서 저장할 경로로 변환하는 함수.
    월, 일에 맞춰 폴더를 생성하고, 날짜 폴더 내에서 파일의 순서에 따라 숫자를 붙여 파일을 저장.
    """
    # 파일 이름에서 고유 식별자를 추출 (예: 13_충남70자1426_1)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 파일 이름에서 맨 마지막에 있는 숫자(언더바 이후)를 제거
    cleaned_file_name = "_".join(file_name.split("_")[:-1])
    
    # 파일 이름 생성 (예: 13_충남70자1426_숫자.csv)
    save_file_name = f"{cleaned_file_name}_{file_counter}.csv"
    
    # 월, 일 폴더 경로
    folder_path = create_date_folder_path(save_base_path, date)
    
    return os.path.join(folder_path, save_file_name)


def count_bus_stops(bus_stop_info_path):
    """
    버스 정류장 정보를 포함한 CSV 파일에서 버스 정류장의 수를 계산하는 함수.
    """
    bus_stop_info = read_csv_with_fallback(bus_stop_info_path)
    return bus_stop_info['STOP_ID'].nunique()  # STOP_ID를 기준으로 정류장 수 계산

def traverse_folders(base_path, save_base_path, bus_stop_count):
    # base_path 안의 모든 하위 폴더들 및 파일들을 순회합니다.
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                process_csv_file(os.path.join(root, file), save_base_path, base_path, bus_stop_count, 1)

if __name__ == "__main__":
    # argparse를 사용하여 경로와 버스 정류장 CSV 파일 경로를 입력받습니다.
    import argparse
    parser = argparse.ArgumentParser(description="Process bus and route data")
    parser.add_argument('--bus_path', type=str, required=True, help="Path to the bus data folder")
    parser.add_argument('--route_path', type=str, required=True, help="Path to the route info CSV file")

    args = parser.parse_args()

    # sort_file 경로를 기반으로 동일한 경로에 sort_day 폴더 생성
    save_base_path = args.bus_path.replace('sort_file', 'sort_day')
    
    # 버스 정류장 수 계산
    bus_stop_count = count_bus_stops(args.route_path)

    # preprocess_data 폴더까지의 경로를 입력받아 하위 폴더를 순회하고, sort_day 폴더에 저장
    traverse_folders(args.bus_path, save_base_path, bus_stop_count)
