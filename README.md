# level2-movie-recommendation-level2-recsys-02
level2-movie-recommendation-level2-recsys-02 created by GitHub Classroom

| |Model|Description|Recall@10| 
|-|---|-------|---|
|1|Metapath2vec|3000 per nodes(inter) * 0.7 + feat emb * 0.3, 1 iter|0.0926|
|2|Metapath2vec|3000 per nodes(inter), 1 iter|0.0884|
|3|Metapath2vec|2000 per nodes, 1 iter|0.0871|
|4|Metapath2vec|1000 per nodes, 1 iter|0.0822|
|5|LightGCN|only ratings|0.0656|
|6|LightGCN|only ratings + 10% popular items|0.0637|
|7|LightGCN|m2v embeddings negative sampling + 10% popular items|0.0139|
|8|Statics-based Model|only ratings|0.1040|
