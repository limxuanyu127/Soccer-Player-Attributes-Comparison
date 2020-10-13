##
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline


# df = pd.read_csv("./cleaned_soccer_data_v1.csv",index_col=0)
df = pd.read_csv("./datasets/cleaned_soccer_data_2016_v2.csv",index_col=0)
print(df.head())
print(df.columns)
exit()
df = df.drop("player_fifa_api_id",1)
binary_features= ["preferred_foot_left",
"preferred_foot_right",
	"attacking_work_rate_high",
    	"attacking_work_rate_low",
        	"attacking_work_rate_medium",
            	"defensive_work_rate_high",
                	"defensive_work_rate_low",
                    	"defensive_work_rate_medium"
]
# df = df.drop(binary_features,1)
# normalise before doing PCA otherwise variance is unbalanced
'''
 Normalization usually means to scale a variable to have a 
 values between 0 and 1, while standardization transforms 
 data to have a mean of zero and a standard deviation of 1
'''
def normalize(df):
    return (df-df.min())/(df.max()-df.min()) #IMPT

df = normalize(df)

# print(df.head())
# scaler = StandardScaler()
# scaler.fit(df)
# X = scaler.transform(df)
print(df.head())
# exit()

# visualization
# https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57

def single_player(df):
    ax = df.iloc[[0]].plot.bar()
    plt.show()

# single_player(df)

def hist_plot_1d(df):
    df.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
            xlabelsize=8, ylabelsize=8, grid=False)    
    plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
    plt.show()

# hist_plot_1d(df)

# may need to drop binary features to see heatmap better
# df = df.drop(binary_features,1)
def corr_mat(df):
    f, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Soccer Player Attributes Correlation Heatmap', fontsize=14)
    plt.show()
corr_mat(df)

# pca? https://builtin.com/data-science/step-step-explanation-principal-component-analysis
# https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e

'''
So, the idea is 10-dimensional data gives you 10 principal components,
 but PCA tries to put maximum possible information in the first component, 
 then maximum remaining information in the second and so on

 The resulting projected data are essentially linear combinations of 
 the original data capturing most of the variance in the data

 centering a variable is subtracting the mean of the variable from each data point 
 so that the new variable's mean is 0; 
 scaling a variable is multiplying each data point 
 by a constant in order to alter the range of the data.

 sklearn PCA only centers the variables. 
 So the sklearn PCA does not feature scale the data beforehand.

 If you don't have any strict constraints, I recommend plotting the cumulative sum 
 of eigenvalues (assuming they are in descending order). 
 If you divide each value by the total sum of eigenvalues 
 prior to plotting, then your plot will show the fraction 
 of total variance retained vs. number of eigenvalues. 
 The plot will then provide a good indication of when you 
 hit the point of diminishing returns 
 (i.e., little variance is gained by retaining additional 
 eigenvalues).

 Reduce dimensions due to the Curse of Dimensionality

 If the PCA display* our K clustering result to be orthogonal or close to, then it is a sign that our clustering 
 is sound , each of which exhibit unique characteristics
'''
# exit()
# pca = PCA(n_components=len(df.columns))
# pca.fit(df)
# plt.plot(pca.explained_variance_)
# plt.show()
# print(pca.explained_variance_) #eigenvalues
# # diminishing returns at 6 PCs
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# from pca import pca #nicer library but bottom part does not work

# # Initialize to reduce the data up to the number of componentes that explains 95% of the variance n_pc=16.
# # model = pca(n_components=0.95)

num_pc=16
pca = PCA(n_components=num_pc)
# pca.fit(df)
df_pca = pca.fit_transform(df)
print(pca.components_) #eigenvectors pca.components_ has shape [n_components, n_features]
print(pca.components_.shape)
print(pca.explained_variance_ratio_)
print(df_pca.shape)

def scatter_pca(df_pca,num_pc):
    fig, axes = plt.subplots(1,num_pc)
    axes[0].scatter(df.iloc[:,0], df.iloc[:,1])
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].set_title('Before PCA')
    for i in range(1,num_pc):
        axes[i].scatter(df_pca[:,i-1], df_pca[:,i])
        axes[i].set_xlabel('PC{}'.format(i))
        axes[i].set_ylabel('PC{}'.format(i+1))
        axes[i].set_title('After PCA')
    plt.show()

scatter_pca(df_pca,num_pc)

def biplot(score,coeff,num_pc,labels=None):
    '''
    score: the projected data
    coeff: the eigenvectors (PCs)
    pcax: pca1 index
    pcay: pca2 index
    '''
    for i in range(0,num_pc-1):
        pca1=i
        pca2=i+1
        xs = score[:,i]
        ys = score[:,i+1]
        n=coeff.shape[0]
        scalex = 1.0/(xs.max()- xs.min())
        scaley = 1.0/(ys.max()- ys.min())
        plt.scatter(xs*scalex,ys*scaley)
        for j in range(n): #feature explain variance
            plt.arrow(0, 0, coeff[j,pca1], coeff[j,pca2],color='r',alpha=0.9) 
            if labels is None:
                plt.text(coeff[j,pca1]* 1.15, coeff[j,pca2] * 1.15, "Var"+str(j+1), color='g', ha='center', va='center')
            else:
                plt.text(coeff[j,pca1]* 1.15, coeff[j,pca2] * 1.15, labels[j], color='g', ha='center', va='center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(pca1+1))
        plt.ylabel("PC{}".format(pca2+1))
        plt.grid()
        plt.show()

# Call the biplot function for any number of PCs
biplot(df_pca, np.transpose(pca.components_),2)

# check if plot is correct
# Var 34 and Var 32 are extremely positively correlated
print(np.corrcoef(df.iloc[:,33], df.iloc[:,31])[1,0])
# Var 36 and Var 37 are negatively correlated
print(np.corrcoef(df.iloc[:,35], df.iloc[:,36])[1,0] )


columns=["PC{}".format(i) for i in range(1,num_pc+1)]
df_pca = pd.DataFrame(df_pca,columns=columns)
print(df_pca.head())
# from pca import pca #nicer library but bottom part does not work

# # Initialize to reduce the data up to the number of componentes that explains 95% of the variance.
# # model = pca(n_components=0.95)

# # Or reduce the data towards 2 PCs
# model = pca(n_components=2)

# # Fit transform
# results = model.fit_transform(df)

# # Plot explained variance
# # fig, ax = model.plot()

# # Scatter first 2 PCs
# # fig, ax = model.scatter()

# # Make biplot with the number of features
# fig, ax = model.biplot(n_feat=4)
# plt.show()
# # combinations
