import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

csv_path = "./winequality-red.csv"
# loading csv data
data = pd.read_csv(csv_path)
PCA_array = PCA(n_components=2).fit_transform(data.iloc[:,:-1])
TSNE_array = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(data.iloc[:,:-1])
print(type(PCA_array))

plt.scatter(PCA_array[:,0], PCA_array[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.show()
