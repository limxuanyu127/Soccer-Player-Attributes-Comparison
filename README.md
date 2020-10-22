# Soccer-Player-Attributes-Comparison
Process of cleaning categorial and numerical data in the European Soccer Dataset (https://www.kaggle.com/hugomathien/soccer?)

## 1. Clean data: OK 
	1. Duplicate rows removes: OK
	2. Add ATK,MID,DEF column: OK (function to be copied from feature_engineering.ipynb) (cleaned_soccer_data_2016_v3_with general_labels)
## 2. Embeddings generation:
	1. PCA embeddings with no detailed positions as features: OK (soccer_player_embeddings_feature_no_labels)
	2. PCA embeddings with detailed positions as features: OK (soccer_player_embeddings_feature_labels)
	3. LDA embeddings: In Progress
	4. NN embeddings: In Progress
## 3. Clustering
- [ ] Clustering Technique Implementation
	- [x] KMeans
	- [x] Hierachical Clustering
	- [x] CURE Clustering 
	- [X] DBSCAN
	- [x] Choosing the best model through the elbow method
- [x] Clustering Evaluation
	- [x] Silhouette Coefficient
	- [x] Visualisations through dendograms, label distribution within cluster, silhouette scores
- [ ] Associative Rule Mining *(WIP) - even tho its not our duty we take one for the team*
- [x] Prove that women are truly the superior gender with our efficiency, brians, personality and looks
- [ ] Waiting for the boys to make us a sandwhich 
## 4. UI/Application: In Progress
