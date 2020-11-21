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

### Run Overall Clustering Evaluation
Source code for the complete evaluation of a dataset/embedding can be found [here](fifa_similarity_search/Clustering/Clustering\ Hyperparameter\ Optimisation.ipynb)
```bash
cd fifa_similarity_search/Clustering/
jupyter notebook Clustering\ Hyperparameter\ Optimisation.ipynb
```
Load your dataset, extract the features you want to fit into the clustering model and you are good to go!

### DBScan Experiment
Source code for DBScan experiment to show k-distance graph can be found [here](fifa_similarity_search/Clustering/DBScan_KDist.ipynb). To access it:
```bash
cd fifa_similarity_search/Clustering/
jupyter notebook DBScan_KDist.ipynb
```

### Associative Rule Mining (ARM)
Source code for ARM experiments can be found [here](fifa_similarity_search/Clustering/association_rule_mining.ipynb). To access it:
```bash
cd fifa_similarity_search/Clustering/
jupyter notebook association_rule_mining.ipynb
```

## 4. UI/Application: In Progress
