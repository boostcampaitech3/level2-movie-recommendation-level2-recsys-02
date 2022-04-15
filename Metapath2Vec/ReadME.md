## 01. 모델 설명 
[[Paper]](https://dl.acm.org/doi/10.1145/3097983.3098036)
- Metapath2vec은 heterogeneous type의 그래프에서 노드의 타입 별로 가능한 metapath를 지정하고, random walk로 노드의 문장을 생성하는 모델
- random walk로 생성한 문장을 skipgram 방법으로 학습시켜 각 노드의 임베딩 계산
- 변형된 MovieLens Dataset에 맞게 두가지의 metapath embedding 학습 진행  
  1) movie -> genre -> movie -> writers -> movie -> directors (영화의 side information 기반 학습)
  2) user -> movie (user 간 interaction 정보 기반 학습)
- Metapath2vec으로 학습한 임베딩을 기반으로 사용자가 평점을 매긴 영화와의 유사한 영화 임베딩을 cosine similarity를 통해 계산
- [[링크]](https://github.com/dmlc/dgl)를 참고하여 코드 구성


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
