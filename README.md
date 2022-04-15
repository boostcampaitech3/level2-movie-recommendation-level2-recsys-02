# level2-movie-recommendation-level2-recsys-02
level2-movie-recommendation-level2-recsys-02 created by GitHub Classroom

| |Model|Description|Recall@10| 
|-|---|-------|---|
|1|FPMC|window size max|0.1180|
|2|Statics-based Model|static + m2v embeddings|0.1165|
|3|FPMC|window size 10|0.1141|
|4|Statics-based Model|static(item popular 0.5, item appear 0.5)|0.1109|
|5|FPMC|only previous data|0.1086|
|6|Statics-based Model|static item appear|0.1040|
|7|BPR|10 Negative Samples|0.1040|
|8|Metapath2vec|3000 per nodes(inter) * 0.7 + feat emb * 0.3, 1 iter|0.0926|
|9|Metapath2vec|3000 per nodes(inter), 1 iter|0.0884|
|10|Metapath2vec|2000 per nodes, 1 iter|0.0871|
|11|BPR|5 Negative Samples|0.0829|
|12|Metapath2vec|1000 per nodes, 1 iter|0.0822|
|13|LightGCN|only ratings|0.0656|
|14|LightGCN|only ratings + 10% popular items|0.0637|
|15|DeepFM|One-hot Encoding + Multi-hot Encoding|0.0565|
|16|DeepFM|One-hot Encoding|0.0223|
|17|LightGCN|m2v embeddings negative sampling + 10% popular items|0.0139|