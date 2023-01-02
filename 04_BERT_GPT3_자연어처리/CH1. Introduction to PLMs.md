## 1.1 Transfer Learning

### **1) Intro**

- Motivations: 다른 데이터를 가지고 다른 사물을 예측하는 vision의 방식 참고 → 다른 도메인의 데이터를 활용할 수도 있다. (공통된 feature 찾기)
- 그럼 NLP에서 공통된 feature는 뭘까?
어려운 단어를 학습할 때, 쉬운 단어를 먼저 많이 학습하고 난 모델이 더 잘 인식할 수 있지 않을까?

### 2) Transfer Learning이란?

- 특정 하나의 문제를 해결하는 데 다른 관련된 문제에서 얻은 지식을 활용한다.
- Big Dataset → Pretraining에서 얻은 weights load
- 해당 weight으로 Target Dataset에 적용한다: Fine-tuning
- 원래는 random하게 initialize된 weight로 target dataset에 적용했지만, 이 dataset의 양이 적을 경우 pretrained된 weight을 활용!

### 3) How to

(1) Set seed weights and train as normal 

: 평소처럼 학습

(2) Fix loaded weights and train unloaded parts

: pretrained weights(freeze) + 달라진 class 개수에 해당하는 initialized weights(ex. softmax layer)

(3) Train with different learning rate on each part

: freeze된 pretrained weights는 낮은 속도로 학습, 새로 initialize한 softmax layer와 같은 신규 weight는 빠른 속도로 학습하도록 다른 learning rate 주기

### 4) Wrap-up

- 대부분의 데이터들은 주요 특징을 공유할 가능성이 있다.
ex. word sense, syntax(NLP), edge, border, 곡면, 직선 등(vision)
- 큰 데이터셋을 통해 미리 훈련한 네트워크를 내 task에 fine-tuning시키자!

## 1.2 Self-supervised Learning

### **1) Self-supervised Learning**

- Unsupervised Learning
    
    : 인간에 의한 label없이, 데이터 x 자체의 내부 표현 또는 feature를 학습
    
    → GAN 등 Generative Modeling, AutoEncoder 등 Representation Learning(벡터화)
    
- Self-supervised Learning
    
    : 데이터 내부 구조를 활용해서 label이 있는 것처럼 학습
    
    → 샘플의 일부정보는 x / 나머지 정보는 y(label) 삼아서 예측
    
    → 이렇게 학습된 모델을 다른 task에 transfer learning해서 지도학습 성능 극대화
    
    ex. 다른 토큰들 학습을 통해 빈칸에 등장할 토큰 예측
    
    → ssl을 통해 좋은 weight param의 seed를 얻어, transfer learning을 통해 한정된 데이터셋에서도 좋은 성능을 얻자!
    
- Contrastive Learning
    
    : 타겟 이미지와 비슷한 x ↔ 비슷하지 않은 x  각각의 거리 차이가 커지도록 학습
    

## 1.3 Introduction to Pretrained Language Models

### **1) PLM**

- Transformer: Attention만을 활용하여 아키텍처 구성
- PLM을 통한 성능 향상
    
    <aside>
    🧐 (1) Feature-based Approach: 더 좋은 embedding을 갖게 하자
    (2) Fine-tuning Approach: 더 좋은 weight parameter seed를 갖게 하자
    (3) Meta-learning Approach(GPT-3): 큰 모델을 다시 학습할 필요 없이, context learning으로 학습 및 추론을 수행
    
    </aside>
    
- Era of PLMs
    
    ![Untitled](../04_BERT_GPT3_%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/images/4-1.png)
    
    parameter가 linear하게 꾸준히 상승 + BERT가 코드 효율로 새로운 패러다임을 불러옴
    

### 2**) Conclusion**

- 타겟 태스크에 따라 transformer모델의 일부를 선택(ex. Encoder or Decoder)
- 많은 unlabeled corpus를 통해 general representation(embedding)학습
- 이후에 target task에 fine-tuning

## 1.4 Downstream Tasks

### 1**) Benchmark Tests**

(1) GLUE: General Language Understanding Evaluation
ex. texutal entailment, text comparison, qa, 감정분석 등

- MNLI: 첫번째 문장과 두번째 문장 간의 관계 classify
metric
- RTE(binary 분류)
- QQP: 주어진 두 문장이 실제로 semantic하게 유사한지
- STS-B: 얼마나 비슷한가를 단계로 나누어 판별
- QNLI:  질문에 대한 답이 잘 이루어졌는가
- SST-2: binary 분류(감정분석)
- CoLA: linguistically acceptable or not

(2) SQUAD 1.1&2.0(Stanford Question Answering Dataset)

→ 벤치마트 테스트 데이터셋을 통해 문제해결능력을 가늠하거나 PLM 성능을 체크

### 2) How to test

- Hugginface 또는 ParlAI 라이브러리에서 테스트코드 제공

```python
from datasets import load_dataset
dataset = load_dataset('squad', split = 'train)
print(', '.join(dataset for dataset in datasets_list))
```

### 3) Korean Benchmark Test

- [Naver sentiment movies corpus v1.0(NSM)](http://github.com/e9t/nsmc)
- [KLUE: Korean Language Understanding Evaluation](http://klue-benchmark.com/tasks)

### 4) Conclusion

- 기계번역이 정복된 후에 nlp의 다른 분야가 연구됨에 따라 PLM의 시대에 앞서 다양한 벤치마크 테스트가 구축됨