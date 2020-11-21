# Soccer-Player-Attributes-Comparison
Process of cleaning categorial and numerical data in the European Soccer Dataset (https://www.kaggle.com/hugomathien/soccer?)

## 1. Clean data: OK 
	1. Duplicate rows removes: OK
	2. Add ATK,MID,DEF column: OK (function to be copied from feature_engineering.ipynb) (cleaned_soccer_data_2016_v3_with general_labels)
## 2. Embeddings generation:
	1. PCA embeddings with no detailed positions as features: OK (soccer_player_embeddings_feature_no_labels_120K)
	2. PCA embeddings with detailed positions as features: OK (soccer_player_embeddings_feature_labels_120K)
	3. LDA embeddings: OK (soccer_player_embeddings_feature_no_labels_LDA_120K)
	4. NN embeddings: In Progress
## 3. Clustering
- [X] Clustering Technique Implementation
	- [x] KMeans
	- [x] Hierachical Clustering
	- [x] CURE Clustering 
	- [X] DBSCAN
	- [x] Choosing the best model through the elbow method
- [x] Clustering Evaluation
	- [x] Silhouette Coefficient
	- [x] Visualisations through dendograms, label distribution within cluster, silhouette scores

Associative Rule Mining (ARM)
Source code for ARM experiments can be found in the jupyer notebook [file](fifa_similarity_search/Clustering/association_rule_mining.ipynb). To access it, use the following command:
```bash
cd fifa_similarity_search/Clustering/
jupyter notebook association_rule_mining.ipynb
```

DBScan Experiment
Source code 
Source code for DBScan experiment to show k-distance graph can be found in the jupyer notebook [file](fifa_similarity_search/Clustering/DBScan_KDist.ipynb). To access it, use the following command:
```bash
cd fifa_similarity_search/Clustering/
jupyter notebook DBScan_KDist.ipynb
```
- [x] Prove that women are truly the superior gender with our efficiency, brians, personality and looks
- [ ] Waiting for the boys to make us a sandwhich 
## 4. UI/Application: In Progress
