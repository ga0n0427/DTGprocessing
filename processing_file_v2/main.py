import subprocess
import argparse

def main():
    # argparse를 사용하여 경로와 버스 경로 CSV 파일 경로를 입력받습니다.
    parser = argparse.ArgumentParser(description="Manage and execute multiple Python scripts for bus data processing")
    parser.add_argument('bus_path', type=str, help="Path to the bus data folder")
    parser.add_argument('route_path', type=str, help="Path to the route info CSV file")
    parser.add_argument('route_name', type=str, help="route number")
    parser.add_argument('output_path', type=str, help="Path to the output CSV file")
    args = parser.parse_args()

    bus_path = args.bus_path
    route_path = args.route_path
    route_name = args.route_name
    output_path = args.output_path
    
    try:
        # calculate_distances.py 실행
        subprocess.run(['python', 'calculate_distances.py', '--route_path', route_path], check=True)
        print("calculate_distances.py 실행 성공")
    except subprocess.CalledProcessError:
        print("calculate_distances.py 실행 중 오류 발생")
        return
    
    try:
        # SearchBus_v.05.py 실행
        subprocess.run(['python', 'SearchBus_v.05_op.py', '--bus_path', bus_path, '--route_path', route_path, '--route_name', route_name], check=True)
        print("SearchBus_v.05.py 실행 성공")
        print(bus_path)
    except subprocess.CalledProcessError:
        print("SearchBus_v.05.py 실행 중 오류 발생")
        return
    
    try:
        # bus_path를 수정하고 processing.py 실행
        bus_path = bus_path.replace('orderbyNumber', 'sort_file')
        subprocess.run(['python', 'processing_v2.py', '--bus_path', bus_path, '--route_path', route_path], check=True)
        print("processing.py 실행 성공")
        print(bus_path)
    except subprocess.CalledProcessError:
        print("processing.py 실행 중 오류 발생")
        return

    try:
        # bus_path를 다시 수정하고 fill_space_num.py 실행
        bus_path = bus_path.replace('sort_file', 'sort_day')
        subprocess.run(['python', 'fill_space_num.py', '--bus_path', bus_path, '--route_path', route_path], check=True)
        print("fill_space_num.py 실행 성공")
    except subprocess.CalledProcessError:
        print("fill_space_num.py 실행 중 오류 발생")
        return

    try:
        # merge_bus_data.py 실행
        subprocess.run(['python', 'merge_bus_data.py', '--bus_path', bus_path, '--route_path', route_path, '--output_path', output_path], check=True)
        print("merge_bus_data.py 실행 성공")
    except subprocess.CalledProcessError:
        print("merge_bus_data.py 실행 중 오류 발생")
        return

if __name__ == "__main__":
    main()
