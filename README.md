## DTG Preprocessing Files

##### This repository contains the DTG preprocessing files. It is composed of six distinct files, each serving a specific purpose in handling and processing DTG and bus route data.

### Overview of Files

#### 1. : The main script that manages the overall flow of the DTG data preprocessing.

calculate_distances: Calculates the distances between bus stops using the Naver API to determine the operational distance between stops.

SearchBus_v.05: Converts bus stop arrival data into route data, allowing for a more structured representation of the bus routes.

processing: Preprocesses incomplete or incorrect parts of the route data before filling in any missing values.

fill_space_num: Fills in the missing values to ensure that as much data as possible is obtained.

merge_bus_data: Merges individual CSV files into a single consolidated file for easier analysis.

### 한글 설명

이 저장소는 DTG 전처리 파일들을 포함하고 있습니다. 총 6개의 파일로 구성되어 있으며, 각 파일은 DTG 및 버스 노선 데이터를 처리하는 데 특정한 역할을 수행합니다.

파일 개요

#### 1. : DTG 데이터 전처리의 전체 흐름을 관리하는 메인 스크립트입니다.

calculate_distances: 네이버 API를 사용하여 버스 정류장 간의 운행 거리를 계산해줍니다.

SearchBus_v.05: 버스 정류장 도착 데이터를 노선 데이터로 변환해줍니다.

processing: 결측치를 채우기 전에 부정확한 노선 데이터의 일부를 전처리합니다.

fill_space_num: 최대한 많은 데이터를 확보하기 위해 결측치를 채워줍니다.

merge_bus_data: 개별 CSV 파일들을 하나의 통합된 파일로 병합합니다.

