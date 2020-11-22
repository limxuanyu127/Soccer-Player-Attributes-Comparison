# Soccer-Player-Attributes-Comparison
Process of cleaning categorial and numerical data in the European Soccer Dataset (https://www.kaggle.com/hugomathien/soccer?)

## 1. Clean data: OK 
	1. Duplicate rows removes: OK
	2. Add ATK,MID,DEF column: OK (function to be copied from feature_engineering.ipynb) (cleaned_soccer_data_2016_v3_with general_labels)

## 2. Embedding Generation
### PCA Experiment
Source code for PCA experiment can be found [here](fifa_similarity_search/embedding_generation/PCA.ipynb). To access it:
```bash
cd fifa_similarity_search/embedding_generation/
jupyter notebook PCA.ipynb
```
### LDA Experiment
Source code for PCA experiment can be found [here](fifa_similarity_search/embedding_generation/LDA.ipynb). To access it:
```bash
cd fifa_similarity_search/embedding_generation/
jupyter notebook LDA.ipynb
```

### DNN Experiment
#### DNN Linear Regression Experiment
Source code for DNN experiment can be found [here](fifa_similarity_search/embedding_generation/dnn_linear_regression.ipynb). To access it:
```bash
cd fifa_similarity_search/embedding_generation/
jupyter notebook dnn_linear_regression.ipynb
```
#### DNN Multi Label Classification Experiment
Source code for DNN Multi Label Classification experiment can be found [here](fifa_similarity_search/embedding_generation/dnn_multi_label_classification.ipynb). To access it:
```bash
cd fifa_similarity_search/embedding_generation/
jupyter notebook dnn_multi_label_classification.ipynb
```
#### DNN Single Label Classification Experiment
Source code for DNN experiment can be found [here](fifa_similarity_search/embedding_generation/dnn_single_label_classification.ipynb). To access it:
```bash
cd fifa_similarity_search/embedding_generation/
jupyter notebook dnn_single_label_classification.ipynb
```

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

## 4. UI/Application (Streamlit)
Source code for application can be found [here](streamlit_app/player_search.ipynb). To access it:
```bash
jupyter notebook player_search.ipynb
```
*NOTE: It is recommended to run the source code on Google Colab or a Linux system, because the Faiss library used is not compatible with Windows*
### Instructions for running:
- In the above source code (*player_search.ipynb*), under the section __Install Packages__, run all cells to install required packages, including *Streamlit*
- Under the section __Generate Streamlit Script__, run the cell to generate *search_script.py*
- Under the section __Run Streamlit Script__
  - Run the first cell to run the script
  - If a remote server/host is used to run the source code, run the second cell (*pyngrok*) to create a tunnel from the remote host
- Access the streamlit app on **http://localhost:8501**, or the output public url if *pyngrok* is used
