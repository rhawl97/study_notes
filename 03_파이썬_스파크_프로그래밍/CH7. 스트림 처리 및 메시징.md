## 1. 스파크 스트리밍 소개

### 1) 스파크 스트리밍 아키텍처

- DStream: RDD에 저장된 데이터의 배치
- Streaming Context
    - Spark Context와 유사
    - 기존 스파크 콘텍스트를 사용하는 스파크 클러스터에 대한 연결
    - DStream을 만들고, 스트리밍 계산 및 변환
    - batchDuration: 스트리밍 데이터가 일괄 처리로 분할되는 시간 간격(초) → 스트리밍 콘텍스트가 인수로 받음
    - ssc.start() / ssc.stop()
    
    ```python
    from pyspark.streaming import StreamingContext
    ssc = StreamingContext(sc,1)
    
    # 데이터 스트림 초기화
    # DStream 변환
    
    ssc.start()
    
    # ssc.stop() or ssc.awaitTermination()
    ```
    

### 2) DStream 소개

- DStream: 연속적인 데이터 스트림에서 생성된 연속적인 RDD 시퀀스
- RDD와 같이 변환, 출력 연산을 지원함

### 3) DStream 소스

- 입력 데이터가 스트리밍콘텍스트 내에 정의됨 (→ RDD 역시 스파크콘텍스트 내의 입력 데이터 소스가 정의됨)
- DStream 생성

```python
from pyspark.streaming import StreamingContext
ssc = StreamingContext(sc, 1)
lines = ssc.socketTextStream('localhost',8881) # hostname, port -> 정의된 TCP 소스에서 DStream을 만든다.
counts = lines.flatMap(lambda line: line.split(" ")).map(lambda word: (word,1)).reduceByKey(lambda a, b: a+b)
counts.pprint()

ssc.start()
ssc.awaitTermination()
```

### 4) DStream 리니지 및 체크포인트

- DAG는 스트리밍콘텍스트에 정의된대로 각 일괄 처리 간격에 체크포인트를 지정함

```python
StreamingContext.checkpoint(directory)  # 주기적으로 특정 DStream의 RDD를 체크포인트로 지정
DStream.checkpoint(interval)            # interval: DStream의 기본 RDD가 체크포인트를 수행한 후의 시간  -> 스트리밍콘텍스트에 설정된 batchDuration의 배수여야 함. 
```

## 2. 슬라이딩 윈도우 연산

### 1) 슬라이딩 윈도우

- 지정된 기간 = 윈도우 길이
- 특정 간격 = 슬라이드 간격
- 슬라이딩 윈도우 연산: 지정된 기간에 걸쳐 DStream 내에서 RDD를 확장하고 특정 간격으로 평가된다.

```python
lines = ssc.socketTextStream('localhost', 8881)  # 데이터 스트림: apple\banana\door\vital\...
## -> apple-banana / door-vital / .. -> (apple,1) (banana,1) (door,1) (vital,1)
windowed = lines.map(lambda word: (word,1)).reduceByKeyAndWindow(lambda x, y: x+y, 2 ,2)   #windowed stream: 마지막 2개의 간격을 2 간격으로 줄인다.
```

- *Dstream*.*window(windowLength, slideInterval) :* Dstream의 지정된 배치에서 새 DStream을 반환한다.
    
    slideIinterval 인수에 지정된 간격마다 새 DStream 객체를 만들고, 지정된 windowlength에 대한 입력 DStream의 요소로 구성된다.
    
    slideIinterval 과 windowlength 모두 스트리밍콘텍스트에 설정된 batchDuration의 배수여야 함