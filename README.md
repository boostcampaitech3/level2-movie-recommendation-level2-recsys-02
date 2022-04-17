# level2-movie-recommendation-level2-recsys-02
level2-movie-recommendation-level2-recsys-02 created by GitHub Classroom

| |Model|Description|Recall@10| 
|-|---|-------|---|
|1|Multi VAE + Multi DAE|11model ensenble(different embedding & dropout)|0.1526
|2|Multi VAE|6model ensenble(different embedding & dropout)|0.1496
|3|Multi DAE|6model ensenble(different embedding & dropout)|0.1464
|4|FPMC|window size max|0.1180|
|5|Statics-based Model|static + m2v embeddings|0.1165|
|6|FPMC|window size 10|0.1141|
|7|Statics-based Model|static(item popular 0.5, item appear 0.5)|0.1109|
|8|FPMC|only previous data|0.1086|
|9|Statics-based Model|static item appear|0.1040|
|10|BPR|10 Negative Samples|0.1040|
|11|Metapath2vec|3000 per nodes(inter) * 0.7 + feat emb * 0.3, 1 iter|0.0926|
|12|Metapath2vec|3000 per nodes(inter), 1 iter|0.0884|
|13|Metapath2vec|2000 per nodes, 1 iter|0.0871|
|14|BPR|5 Negative Samples|0.0829|
|15|Metapath2vec|1000 per nodes, 1 iter|0.0822|
|16|LightGCN|only ratings|0.0656|
|17|LightGCN|only ratings + 10% popular items|0.0637|
|18|DeepFM|One-hot Encoding + Multi-hot Encoding|0.0565|
|19|DeepFM|One-hot Encoding|0.0223|
|20|LightGCN|m2v embeddings negative sampling + 10% popular items|0.0139|
