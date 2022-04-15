## 01. 모델 설명 
[[Paper]](https://arxiv.org/abs/2002.02126)
- LightGCN은 임베딩 노드의 파라미터를 곱하지 않고 단순히 가중합하여 경량화한 모델
- 그래프 내 노드 Convolution 과정에서 이웃 노드의 임베딩을 단순 가중합만 하고, 그 외의 학습 파라미터는 존재하지 않아 연산량 감소
- 레이어가 깊어질수록 강도가 약해질 것이라는 아이디어 → 모델 단순화
- [[링크]](https://medium.com/stanford-cs224w/lightgcn-for-movie-recommendation-eb6d112f1e8)를 참고하여 dgl로 코드를 리팩토링 진행


## 02. Requirements
```
dgl >= 0.8.x
torch >= 1.9.0
```
dgl은 [여기](https://www.dgl.ai/pages/start.html)서 설치 가능합니다.


## 03. 실행 방법
```
python train.py
```
