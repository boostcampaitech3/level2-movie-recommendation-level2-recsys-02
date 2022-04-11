| |Model|Description|Recall@10| 
|-|---|-------|---|
|1|FPMC|window size max|0.1180|
|2|FPMC|window size 10|0.1141|
|3|FPMC|only previous data|0.1086|
|4|BPR|10 Negative Samples|0.1040|
|5|BPR|5 Negative Samples|0.0829|
|6|DeepFM|One-hot Encoding + Multi-hot Encoding|0.0565|
|7|DeepFM|One-hot Encoding|0.0223|
|?|CBPR|Embedding + BPR|-|
|?|CBPR|FM + BPR|-|
|?|CBPR|DeepFM + BPR|-|
|?|CBPR|m2v + BPR|-|
|?|GRU4REC||-|