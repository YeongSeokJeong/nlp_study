# Spelling Correction Model

## 논문

제목 : [A spelling correction model for end-to-end speech recognition](https://paperswithcode.com/paper/a-spelling-correction-model-for-end-to-end)

출처 : Accepted to ICASSP 2019

## 동기

 음성인식에서 End-to-End 모델의 경우 음성-문장의 쌍을 이룬 데이터가 필요하다. 이로 인해 ASR 모델의 경우 LM(Language Model)에 비해 적은 데이터를  학습하게 된다.  이는 데이터 적은 빈도로 존재하는 단어, 고유 명사에 대한 성능을 감소시킨다.  이런 데이터의 한계를 극복하기 위해 많은 연구들이 RNN과LM을 결합한 형태의 모델을 고안하였다. 하지만 이런 결합에도 불구하고 드문 단어에 대한 성능 저하와 고유명사로 인한 성능 저하의 문제는 여전히 존재 하였다. 

---

### 논문에서 사용한 데이터

데이터 : LibriSpeech

Input data : LibriSpeech데이터를 input으로 하여 LAS에 의해 만들어진 output text

​                      이를 Beam Search를 사용하여 8개의 후보 문장을 만듦

​                      8개의 문장 가운데 랜덤으로 문장을 뽑아 input text로 선정

output data : 실제 값

---

### 논문에서 제안하는 모델

![](img\model architecture.jpg)

기존 모델과 다른점

- Encoder와 Decoder에서 잔차 연결(residual connection)을 함
- LSTM의 각 Cell에서 Layer Normalization을 함
- Multi-headed Attention의 경우 additive attention을 수행
- Multi-headed Attention의 경우 Projection layer 의 결과와 decoder의 최저단의 LSTM 결과를 사용
- Multi-headed Attention의 경우 decoder의 결과와 concatenation된다.

---

### 실험 결과

기존 모델에 비해 약 29%정도 성능이 좋아진 것을 확인