#### 1. Load Item Embedding
```
with open('./m2v_item_emb.pkl', 'rb') as f :
    m2v_item_emb = pickle.load(f)
```

#### 2. Load Item Embedding Index Dict
```
with open('./m2v_item_index.pkl', 'rb') as f :
    m2v_item_index = pickle.load(f)
```

#### How To Use
m2v_item_index에서 해당 아이템의 인덱스를 불러올 수 있습니다.  

- m2v_item_index 예시)
```
train_rating.csv 파일에서 item id 1 = 'i1'   
{'i8973': 0,
 'i1097': 1,
 'i3363': 2,
 'i260': 3,
 'i3504': 4,
 ...
 }
```
이 인덱스를 통해 해당 아이템의 임베딩이 m2v_item_emb의 몇 번째 임베딩인지 확인할 수 있습니다.
