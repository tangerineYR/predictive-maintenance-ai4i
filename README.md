<div align="center">

# AI4I Equipment Failure Classification

3학년 2학기 사물기계학습 과목에서 수행한 개인 프로젝트입니다. UCI AI4I 2020 합성 설비 데이터에서 온도, 회전 속도, 토크, 공구 마모 시간, 제품 유형을 사용해 **같은 관측 행의 고장 여부**를 Random Forest로 분류했습니다. 입력에 정답이 섞이지 않도록 고장 유형 열을 제외하고, 교차검증으로 모델 설정을 고른 뒤 고장 탐지 성능을 기준 모델과 비교했습니다.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-555555)](#data-and-training)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Python 검사](https://github.com/tangerineYR/predictive-maintenance-ai4i/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tangerineYR/predictive-maintenance-ai4i/actions/workflows/ci.yml)

[저장된 평가 지표](results/metrics.json) · [평가 그림](results/evaluation.png) · [실행 방법](#reproduction)

</div>

## Test Results

테스트 2,000건 가운데 실제 고장은 68건이었습니다. Random Forest는 **고장 41건을 탐지**해 재현율 **60.29%** 를 기록했습니다. 고장 27건을 놓쳤고, 정상 1,932건 중 21건을 고장으로 분류했습니다.

| 지표 | Random Forest | 항상 정상으로 분류 |
| --- | ---: | ---: |
| 정확도 | 0.9760 | 0.9660 |
| 고장 정밀도 | 0.6613 | 0.0000 |
| 고장 재현율 | 0.6029 | 0.0000 |
| 고장 F1 | 0.6308 | 0.0000 |
| ROC-AUC | 0.9565 | 0.5000 |
| 평균 정밀도(AP) | 0.6745 | 0.0340 |

항상 정상으로 분류해도 정확도는 96.60%입니다. Random Forest와의 정확도 차이는 **1.00%p**입니다. Random Forest가 고장으로 분류한 62건 중 실제 고장은 41건이어서 정밀도는 **66.13%** 였습니다.

![테스트 혼동행렬, ROC 곡선 및 Random Forest 특성 중요도](results/evaluation.png)

그림의 혼동행렬에서 행은 실제값, 열은 예측값입니다. 오른쪽의 특성 중요도는 트리 분할에서 불순도를 줄이는 데 쓰인 정도를 보여줍니다.

### Precision-Recall Metrics

표의 AP는 scikit-learn의 `average_precision_score`로 계산한 값입니다. 기준 모델은 고장 점수를 항상 0으로 매기므로 AP는 테스트 고장 비율인 `68 / 2,000 = 0.0340`입니다.

결과 JSON의 `pr_auc`는 정밀도–재현율 곡선의 사다리꼴 면적입니다. Random Forest는 **0.6719**이며, 기준 모델의 **0.5170**은 재현율 1에서의 정밀도 0.034와 재현율 0에서의 정밀도 끝점 1을 선형으로 연결해 생긴 값입니다. 기준 모델의 고장 판정은 0건이고, 이때 정밀도는 `zero_division=0` 설정에 따라 0으로 기록합니다. [평균 정밀도 계산](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.average_precision_score.html) · [곡선 끝점 정의](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.precision_recall_curve.html)

## Data and Training

전체 10,000건 중 고장은 339건(3.39%)입니다. CSV에서 입력과 정답에 필요한 열을 검사한 뒤, 허용한 6개 원본 특성만 모델에 전달합니다.

| 구분 | 변수와 처리 |
| --- | --- |
| 수치형 입력 5개 | 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모 시간에 `StandardScaler` 적용 |
| 범주형 입력 1개 | 제품 유형 `Type`(`L`, `M`, `H`)에 원-핫 인코딩 적용 |
| 정답 | `Machine failure`의 `1`은 고장, `0`은 정상 |
| 입력에서 제외 | 식별자 `UDI`·`Product ID`와 고장 유형 라벨 `TWF`·`HDF`·`PWF`·`OSF`·`RNF` |

1. 시드 `42`를 사용한 층화 무작위 분할로 훈련 8,000건과 테스트 2,000건을 나눕니다. 고장은 각각 271건, 68건입니다.
2. 훈련 데이터 안에서 층화 5겹 교차검증을 실행합니다. 각 훈련 폴드에서 전처리와 `class_weight="balanced"`를 적용한 Random Forest를 학습합니다. 트리 수는 100개로 고정하고 최대 깊이 `10`, `20`, `None`을 비교합니다.
3. 고장 클래스의 평균 검증 F1이 가장 높은 설정을 훈련셋 전체로 다시 학습합니다. 공개 실험에서 선택된 최대 깊이는 **10**, 평균 검증 F1은 **0.5924**입니다.
4. 설정 선택이 끝나면 같은 테스트셋에서 Random Forest와 기준 모델을 평가합니다. 표의 분류 지표는 각 모델의 기본 `predict()` 판정으로, ROC-AUC와 AP는 고장 점수로 계산합니다.

수치 표준화는 기존 과제의 전처리 구성을 유지한 선택입니다. 성능 비교의 단위는 가중치를 포함한 Random Forest 파이프라인과 항상 정상으로 분류하는 기준 모델입니다. [학습·평가 코드](smart_factory_equipment_failure_pred.py)

## Evaluation Scope

고장 라벨과 센서 값은 같은 관측 행에 있습니다. 보고한 지표의 범위는 이 합성 데이터에서 한 번의 층화 무작위 분할로 수행한 현재 고장 분류입니다. 실제 설비의 사전 경보나 잔여 수명 예측으로 범위를 넓히려면 시간 순서와 설비 단위를 분리한 검증이 필요합니다. 운영 시점에 사용할 판정 임계값도 미탐·오탐 비용에 맞춰 별도로 평가해야 합니다. [UCI 변수 설명](https://archive.ics.uci.edu/dataset/601/ai4i)

## Reproduction

Python 3.11로 실행합니다. 저장된 실험은 Python 3.11.3에서 생성했으며, 직접 의존성 버전은 `pyproject.toml`에 지정되어 있습니다. 먼저 저장소를 복제하고 가상환경을 만듭니다. [패키지 설정](pyproject.toml)

~~~bash
git clone https://github.com/tangerineYR/predictive-maintenance-ai4i.git
cd predictive-maintenance-ai4i
python -m venv .venv
~~~

### Windows PowerShell

~~~powershell
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m smart_factory_equipment_failure_pred --output-dir ../ai4i-results-rerun
~~~

### Linux and macOS

~~~bash
.venv/bin/python -m pip install .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m smart_factory_equipment_failure_pred --output-dir ../ai4i-results-rerun
~~~

전체 실험 명령은 UCI에서 원본 CSV를 읽습니다. 저장소 옆 폴더에 `metrics.json`과 `evaluation.png`를 생성합니다. 같은 명령을 다시 실행할 때에는 별도의 출력 경로를 지정해 실행별 결과를 보존할 수 있습니다.

CSV를 이미 내려받았다면 `--data`로 경로를 지정할 수 있습니다. 아래는 `ai4i2020.csv`가 저장소 옆 폴더에 있을 때의 PowerShell 예시입니다.

~~~powershell
.\.venv\Scripts\python.exe -m smart_factory_equipment_failure_pred --data ../ai4i2020.csv --output-dir ../ai4i-local-results
~~~

결과 JSON에서 데이터 행 수와 고장 수, 분할 설정, 선택된 모델 설정과 라이브러리 버전을 확인할 수 있습니다. 혼동행렬로 정확도·정밀도·재현율·F1을 계산할 수 있으며, 개별 고장 점수가 필요한 ROC-AUC와 AP는 실험을 다시 실행해 확인합니다. [공개 결과 JSON](results/metrics.json)

## Tests and CI

Ubuntu·Python 3.11 환경의 GitHub Actions가 패키지를 설치하고 오프라인 단위 테스트 5개를 실행합니다. 작은 합성 표본으로 필수 열과 입력값 검증, 식별자·고장 유형 열의 제외, 분류기 학습과 기준 모델 지표를 검사합니다. 다운로드 실패 테스트는 네트워크 오류를 모의해 원인 예외가 보존되는지 확인합니다. UCI 원본에 대한 교차검증과 파일 생성은 위 재현 명령으로 실행합니다. [테스트 코드](tests/test_pipeline.py) · [CI 실행 이력](https://github.com/tangerineYR/predictive-maintenance-ai4i/actions/workflows/ci.yml)

## Dataset Attribution

*AI4I 2020 Predictive Maintenance Dataset* (2020), UCI Machine Learning Repository. 원본 CSV에서 수치형 열 5개와 제품 유형을 입력으로 사용했습니다. 데이터셋은 CC BY 4.0으로 제공됩니다. [UCI 데이터셋](https://archive.ics.uci.edu/dataset/601/ai4i) · [DOI](https://doi.org/10.24432/C5HS5C) · [데이터셋 라이선스](https://creativecommons.org/licenses/by/4.0/)