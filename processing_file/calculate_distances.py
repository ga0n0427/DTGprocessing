import requests
import pandas as pd
import time
import argparse


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

# 네이버 Directions5 API를 호출하여 거리 계산 함수
def get_distance_via_naver(lat1, lng1, lat2, lng2):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    params = {
        'start': f'{lng1},{lat1}',
        'goal': f'{lng2},{lat2}',
        'option': 'trafast'  # 빠른 경로로 설정
    }
    headers = {
        'X-NCP-APIGW-API-KEY-ID': client_id,
        'X-NCP-APIGW-API-KEY': client_secret
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            # 경로가 있는 경우 거리 반환 (미터 단위)
            return data['route']['trafast'][0]['summary']['distance']
        except (KeyError, IndexError):
            return None
    else:
        print(f"API 요청 실패: {response.status_code}")
        return None


parser = argparse.ArgumentParser(description="Process bus and route data")
parser.add_argument('--route_path', type=str, required=True, help="Path to the route info CSV file")
args = parser.parse_args()
# CSV 파일 읽기 (예: '/path/to/bus_route_specific_13.csv')
#file_path = 'F:/Data/data/bus_route_specific_13.csv'
df = read_csv_with_fallback(args.route_path)

# 이전 정류장과 다음 정류장 간의 거리를 계산하여 previous와 next 열에 삽입
df['previous'] = None
df['next'] = None
df['MASK_SELECTED'] = 0  # 초기값은 NaN으로 설정

for i in range(len(df)):  # i=0부터 시작
    # 이전 정류장과의 거리 계산 (첫 번째 정류장은 previous를 계산하지 않음)
    if i > 0:
        lat1, lng1 = df.at[i-1, 'LAT'], df.at[i-1, 'LNG']
        lat2, lng2 = df.at[i, 'LAT'], df.at[i, 'LNG']
        df.at[i, 'previous'] = get_distance_via_naver(lat1, lng1, lat2, lng2)
    
    # 다음 정류장과의 거리 계산 (마지막 정류장은 계산하지 않음)
    if i < len(df) - 1:
        lat2, lng2 = df.at[i, 'LAT'], df.at[i, 'LNG']
        lat3, lng3 = df.at[i+1, 'LAT'], df.at[i+1, 'LNG']
        df.at[i, 'next'] = get_distance_via_naver(lat2, lng2, lat3, lng3)
    
    # API 호출 사이에 잠시 대기 (속도 제한을 피하기 위해)
    time.sleep(1)
    
    df.iloc[i, df.columns.get_loc('MASK_SELECTED')] = i + 1  # MASK_SELECTED 값 설정

# 결과 저장
output_path = args.route_path
df.to_csv(output_path, index=False)

print(f"거리 계산이 완료되었습니다. 결과가 {output_path}에 저장되었습니다.")


