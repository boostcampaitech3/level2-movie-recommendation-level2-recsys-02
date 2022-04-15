# level2-movie-recommendation-level2-recsys-02
level2-movie-recommendation-level2-recsys-02 created by GitHub Classroom

| |Model|Description|Recall@10| 
|-|---|-------|---|
|1|Statics-based Model|static + m2v embeddings|0.1165|
|2|Statics-based Model|static(item popular 0.5, item appear 0.5)|0.1109|
|3|Statics-based Model|static item appear|0.1040|
|4|Metapath2vec|3000 per nodes(inter) * 0.7 + feat emb * 0.3, 1 iter|0.0926|
|5|Metapath2vec|3000 per nodes(inter), 1 iter|0.0884|
|6|Metapath2vec|2000 per nodes, 1 iter|0.0871|
|7|Metapath2vec|1000 per nodes, 1 iter|0.0822|
|8|LightGCN|only ratings|0.0656|
|9|LightGCN|only ratings + 10% popular items|0.0637|
|10|LightGCN|m2v embeddings negative sampling + 10% popular items|0.0139|

