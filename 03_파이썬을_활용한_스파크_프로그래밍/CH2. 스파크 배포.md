## 1. 스파크 배포 모드

### 0) Intro

- 배포모드 = 스파크 런타임 아키텍처 구현
- 컴퓨팅 클러스터의 노드에서 리소스를 관리하는 방식이 다름

### 1) 로컬 모드

- 모든 스파크 프로세스가 단일 시스템에서 실행됨
- 로컬 시스템의 코어 수를 임의로 선택해 사용.
- 로컬 모드 spark submit
    
    ```bash
    $SPARK_HOME/bin/spark-submit \
    --class org.apache.spark.examples.SparkPi \
    --master local \
    $SPARK_HOME/examples/jars/spark-examples*.jar 10
    ```
    
- local[*]: 코어 수 지정

### 2) Standalone 모드

- 다중 노드 클러스터에서 외부 스케줄러가 필요 x.
- URI 스키마로 지정된 호스트+포트와 함께 스파크 standalone 클러스터에 제출

### 3) YARN에서의 스파크

- 가장 일반적인 배포 방법: 하둡과 함께 제공되는 YARN 리소스 관리 프레임워크 사용
- YARN을 스케줄러로 사용할 때 cluster와 client 라는 2개의 클러스터 배포 모드 중 하나 사용
    
    <aside>
    
    **Cluster 모드란?**
    드라이버 프로세스가 Cluster 내의 Application Master 에서 실행되는 모드
    driver가 YARN 컨테이너에서 동작하기 때문에 driver에 장애가 발생할 경우 다른 노드에서 driver가 재실행
    주로 Production 환경에 적합
    
    **Client 모드란?**
    드라이버 프로세스가 Cluster 와 관련 없는 다른 외부 서버(ex, 내 방 데탑)에서 실행되는 모드
    주로 개발과정에서 대화형 디버깅을 할 때 사용(스파크 쉘)
    따로 지정하지 않으면 기본으로 선택되는 모드임
    
    </aside>