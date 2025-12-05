import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
import scipy.stats
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import time
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error, adjusted_mutual_info_score, completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

plt.rc('font', size=14)  # 기본 폰트 크기
plt.rc('axes', labelsize=14)  # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=12)  # 범례 폰트 크기
plt.rc('figure', titlesize=18)

data1 = pd.read_csv('data/wisc_bc_data.csv')
data1n = data1.values.copy()
num_row, num_col = np.shape(data1n)
for i in range(num_row):
    if data1n[i, 0] == 'B':
        data1n[i, 0] = 1
    else:
        data1n[i, 0] = 0
x1 = data1n[:, 1:]
minmax = preprocessing.MinMaxScaler()
x1 = minmax.fit_transform(x1)
x1 = x1.astype(float)
y1 = data1n[:, 0]
y1 = y1.astype(int)

data2 = pd.read_csv('data/Wifi-localization.csv')
data2n = data2.values.copy()
x2 = data2n[:, 0:-1]
minmax = preprocessing.MinMaxScaler()
x2 = minmax.fit_transform(x2)
x2 = x2.astype(float)
y2 = data2n[:,-1]
y2 = y2.astype(int)
N=50

# Part1-1. KMeans algorithm for Breast
sse_kmeans = []
for i in range(2, N):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x1)
    sse_kmeans.append(kmeans.inertia_)
plt.figure(1)
plt.plot(np.arange(2,N,1),sse_kmeans, label='k-means', marker='o',)
plt.grid()
plt.title('Inertia of K-means')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('fig1_kmeans_inertia_breast.png')
plt.close()

# Code reference from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#
plt.figure(2)
range_clusters=np.arange(7,10,1)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x1) + (i + 1) * 10])
    km = KMeans(n_clusters=i, random_state=42)
    km_labels = km.fit_predict(x1)
    sample_sil_values = silhouette_samples(x1, km_labels)
    km_silhouette_avg = silhouette_score(x1, km_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[km_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=km_silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(km_labels.astype(float) / i)
    ax2.scatter(x1[:, 0], x1[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = km.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle("Silhouette analysis for KM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig2-%d_silhouette_breast.png' %i)
    plt.close()

kmeans_labels1 = KMeans(n_clusters=7, random_state=42).fit_predict(x1)
print('No Dim.red. K-means silhouette score for B-cancer is :', silhouette_score(x1, kmeans_labels1, metric='euclidean'))
print('No Dim.red. K-means completeness score for B-cancer is :',  completeness_score(y1, kmeans_labels1))

# Part1-2. KMeans algorithm for Wifi
sse_kmeans = []
for i in range(2, N):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x2)
    sse_kmeans.append(kmeans.inertia_)
plt.figure(3)
plt.plot(np.arange(2,N,1),sse_kmeans, label='k-means', marker='o',)
plt.grid()
plt.title('Inertia of K-means')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('fig3_kmeans_inertia_wifi.png')
plt.close()

plt.figure(4)
range_clusters=np.arange(4,7,1)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x2) + (i + 1) * 10])

    km = KMeans(n_clusters=i, random_state=42)
    km_labels = km.fit_predict(x2)
    sample_sil_values = silhouette_samples(x2, km_labels)
    km_silhouette_avg = silhouette_score(x2, km_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[km_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=km_silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(km_labels.astype(float) / i)
    ax2.scatter(x2[:, 0], x2[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    # Labeling the clusters
    centers = km.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle("Silhouette analysis for KM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig4-%d_silhouette_wifi.png' %i)
    plt.close()

kmeans_labels2 = KMeans(n_clusters=4, random_state=42).fit_predict(x2)
print('No Dim.red. K-means silhouette score for Wifi is :', silhouette_score(x2, kmeans_labels2, metric='euclidean'))
print('No Dim.red. K-means completeness score for Wifi is :',  completeness_score(y2, kmeans_labels2))

#Part1-3. EM algoritm for Breast
bic_em = []
for i in range(2, N):
    em = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em.fit(x1)
    bic_em.append(em.bic(x1))
plt.figure(5)
plt.plot(np.arange(2,N,1),bic_em, marker='o',)
plt.title('BIC score for EM without dimension reduction')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig5_EM_Bic_score_before_reduction_breast.png')
plt.close()

# Code reference from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#
plt.figure(6)
range_clusters=np.arange(2,5,1)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x1) + (i + 1) * 10])

    em = GaussianMixture(n_components=i, random_state=42)
    em_labels = em.fit_predict(x1)
    sample_sil_values = silhouette_samples(x1, em_labels)
    em_silhouette_avg = silhouette_score(x1, em_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[em_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=em_silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(em_labels.astype(float) / i)
    ax2.scatter(x1[:, 0], x1[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = np.zeros((i, x1.shape[1]))
    #centers = em.cluster_centers_
    for m in range(i):
        density = scipy.stats.multivariate_normal(cov=em.covariances_[m], mean=em.means_[m]).logpdf(x1)
        centers[m, :] = x1[np.argmax(density)]
        # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)

    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle("Silhouette analysis for EM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig6-%d_silhouette_breast.png' %i)
    plt.close()

em_labels1 = GaussianMixture(n_components=2, random_state=42).fit_predict(x1)
print('No Dim.red. EM silhouette score for B-cancer is :', silhouette_score(x1, em_labels1, metric='euclidean'))
print('No Dim.red. EM completeness score for B-cancer is :',  completeness_score(y1, em_labels1))

#Part1-4. EM algoritm for Wifi

bic_em = []
for i in range(2, N):
    em = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em.fit(x2)
    bic_em.append(em.bic(x2))
plt.figure(7)
plt.plot(np.arange(2,N,1),bic_em, marker='o',)
plt.title('BIC score for EM without dimension reduction')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig7_EM_Bic_score_before_reduction_wifi.png')
plt.close()

# Code reference from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#
plt.figure(8)
range_clusters=np.arange(4, 7, 1)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x2) + (i + 1) * 10])

    em = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em_labels = em.fit_predict(x2)
    sample_sil_values = silhouette_samples(x2, em_labels)
    em_silhouette_avg = silhouette_score(x2, em_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[em_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=em_silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(em_labels.astype(float) / i)
    ax2.scatter(x2[:, 0], x2[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = np.zeros((i, x2.shape[1]))
    #centers = em.cluster_centers_
    for m in range(i):
        density = scipy.stats.multivariate_normal(cov=em.covariances_[m], mean=em.means_[m]).logpdf(x2)
        centers[m, :] = x2[np.argmax(density)]
        # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)

    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle("Silhouette analysis for EM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig8-%d_silhouette_breast.png' %i)
    plt.close()

em_labels2 = GaussianMixture(n_components=5, random_state=42).fit_predict(x2)
print('No Dim.red. EM silhouette score for wifi is :', silhouette_score(x2, em_labels2, metric='euclidean'))
print('No Dim.red. EM completeness score for wifi is :',  completeness_score(y2, em_labels2))


### Part2. Dimension reduction
#Part2-1-1. PCA
N=x1.shape[1]
pca=PCA(random_state=42).fit(x1)
x_range=np.arange(1,N+1)
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(9)
bar=plt.bar(x_range, pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual variance')
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.2f'%height, ha='center', va='bottom', size=8)
plt.plot(x_range, cum_var, marker='o', label='cumulative variance')
plt.ylim(0.0, 1.1)
plt.axhline(y=0.9, color='r', linestyle='--', linewidth=0.5)  #90% select
plt.text(15, 0.9, '90% point of variance', color='r', fontsize=12)
plt.xlabel('Number of clusters')
plt.ylabel('variance')
plt.title('Decision: Number of components for PCA')
plt.grid()
plt.savefig('fig9_pca_breast.png')
plt.close()

#Part2-1-2. PCA
N=x2.shape[1]
pca=PCA(random_state=42).fit(x2)
x_range=np.arange(1,N+1)
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(10)
bar=plt.bar(x_range, pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual variance')
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.2f'%height, ha='center', va='bottom', size=8)
plt.plot(x_range, cum_var, marker='o', label='cumulative variance')
plt.ylim(0.0, 1.1)
plt.axhline(y=0.9, color='r', linestyle='--', linewidth=0.5)  #90% select
plt.text(5, 0.9, '90% point of variance', color='r', fontsize=12)
plt.xlabel('Number of clusters')
plt.ylabel('variance')
plt.title('Decision: Number of components for PCA')
plt.grid()
plt.savefig('fig10_pca_wifi.png')
plt.close()

#Part2-2-1. ICA
N=x1.shape[1]
kur_values=[]
for i in range(1, N):
    x_ica=FastICA(n_components=i, random_state=42, whiten='unit-variance').fit_transform(x1)
    k=kurtosis(x_ica)
    kur_values.append(np.mean(k)/i)
x_range=np.arange(1,N)
plt.figure(11)
plt.plot(x_range, kur_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Kurtosis values')
plt.title('Decision: Number of components for ICA')
plt.grid()
plt.savefig('fig11_ica_breast.png')
plt.close()

#Part2-2-2. ICA
N=x2.shape[1]
kur_values=[]
for i in range(1, N):
    x_ica=FastICA(n_components=i, random_state=42, whiten='unit-variance').fit_transform(x2)
    k=kurtosis(x_ica)
    kur_values.append(np.mean(k)/i)
x_range=np.arange(1,N)
plt.figure(12)
plt.plot(x_range, kur_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Kurtosis values')
plt.title('Decision: Number of components for ICA')
plt.grid()
plt.savefig('fig12_ica_breast.png')
plt.close()

#Part2-3-1. RP
N=x1.shape[1]
test=100
mse_t=np.zeros((N, test))
for i in range(1, N+1):
    for j in range(test):
        err=0
        rp=GaussianRandomProjection(n_components=i, compute_inverse_components=True)
        x_rp=rp.fit_transform(x1)
        err=mean_squared_error(x1, rp.inverse_transform(x_rp))   # n samples X n components
        mse_t[i-1, j] = (err/i)   # divided by components
x_range=np.arange(1,N+1)
plt.figure(13)
plt.plot(x_range, np.mean(mse_t, axis=1), marker='o')
plt.fill_between(x_range,np.mean(mse_t, axis=1) - np.std(mse_t, axis=1), np.mean(mse_t, axis=1) + np.std(mse_t, axis=1), alpha=0.3)
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('Decision: Number of components for RP')
plt.grid()
plt.savefig('fig13_rp_breast.png')
plt.close()

#Part2-3-2. RP
N=x2.shape[1]
test=100
mse_t=np.zeros((N, test))
for i in range(1, N+1):
    for j in range(test):
        err=0
        rp=GaussianRandomProjection(n_components=i, compute_inverse_components=True)
        x_rp=rp.fit_transform(x2)
        err=mean_squared_error(x2, rp.inverse_transform(x_rp))   # n samples X n components
        mse_t[i-1, j] = (err/i)   # divided by components
x_range=np.arange(1,N+1)
plt.figure(14)
plt.plot(x_range, np.mean(mse_t, axis=1), marker='o')
plt.fill_between(x_range,np.mean(mse_t, axis=1) - np.std(mse_t, axis=1), np.mean(mse_t, axis=1) + np.std(mse_t, axis=1), alpha=0.3)
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('Decision: Number of components for RP')
plt.grid()
plt.savefig('fig14_rp_wifi.png')
plt.close()

plt.figure(151)
N=x1.shape[1]
err1=[]

for i in range(1, N):
    lle1=LocallyLinearEmbedding(n_neighbors=i, n_components=7, random_state=42, eigen_solver='dense', method='standard')
    x_lle1=lle1.fit_transform(x1)
    err1.append(lle1.reconstruction_error_)
x_range=np.arange(1,N, 1)
plt.plot(x_range, err1, marker='o')
plt.xlabel('Number of neighbors')
plt.ylabel('Reconstruction error')
plt.title('Decision: Number of neighbors for LLE')
plt.grid()
plt.tight_layout()
plt.savefig('fig15-1_lle_breast_neighbors.png')
plt.close()

plt.figure(152)
N=x1.shape[1]
err1=[]
for i in range(1, N):
    lle1=LocallyLinearEmbedding(n_neighbors=11, n_components=i, random_state=42, eigen_solver='dense', method='standard')
    x_lle1=lle1.fit_transform(x1)
    err1.append(lle1.reconstruction_error_)
x_range=np.arange(1,N, 1)
plt.plot(x_range, err1, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction error')
plt.title('Decision: Number of components for LLE')
plt.grid()
plt.tight_layout()
plt.savefig('fig15-2_lle_breast_components.png')
plt.close()

plt.figure(161)
err1=[]
for i in range(1, 30):
    lle1=LocallyLinearEmbedding(n_neighbors=i, n_components=3, random_state=42, eigen_solver='dense', method='standard')
    x_lle1=lle1.fit_transform(x2)
    err1.append(lle1.reconstruction_error_)
x_range=np.arange(1,30, 1)
plt.plot(x_range, err1, marker='o')
plt.xlabel('Number of neighbors')
plt.ylabel('Reconstruction error')
plt.title('Decision: Number of neighbors for LLE')
plt.grid()
plt.tight_layout()
plt.savefig('fig16-1_lle_wifi_neighbors.png')
plt.close()

plt.figure(162)
N=x2.shape[1]
err1=[]
for i in range(1, N):
    lle1=LocallyLinearEmbedding(n_neighbors=19, n_components=i, random_state=42, eigen_solver='dense', method='standard')
    x_lle1=lle1.fit_transform(x2)
    err1.append(lle1.reconstruction_error_)
x_range=np.arange(1, N, 1)
plt.plot(x_range, err1, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction error')
plt.title('Decision: Number of components for LLE')
plt.grid()
plt.tight_layout()
plt.savefig('fig16-2_lle_wifi_components.png')
plt.close()


#Part3-1-1. KM + PCA
sse_kmeans1 = []
sse_kmeans2 = []
x_pca1 = PCA(n_components=4, random_state=42).fit_transform(x1)
x_pca2 = PCA(n_components=3, random_state=42).fit_transform(x2)
for i in range(2, N):
    kmeans1 = KMeans(n_clusters=i, random_state=42)
    kmeans1.fit(x_pca1)
    sse_kmeans1.append(kmeans1.inertia_)
    kmeans2 = KMeans(n_clusters=i, random_state=42)
    kmeans2.fit(x_pca2)
    sse_kmeans2.append(kmeans2.inertia_)
plt.figure(171)
plt.plot(np.arange(2,N,1),sse_kmeans1, label='B-cancer', marker='o',)
plt.grid()
plt.title('Inertia of K-means after PCA: B-cancer')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('fig17-1_kmeans_inertia_pca_breast.png')
plt.close()

plt.figure(172)
plt.plot(np.arange(2,N,1),sse_kmeans2, label='Wifi', marker='o',)
plt.grid()
plt.title('Inertia of K-means after PCA: Wifi')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('fig17-2_kmeans_inertia_pca_wifi.png')
plt.close()

kmeans_labels1_pca = KMeans(n_clusters=7, random_state=42).fit_predict(x_pca1)
kmeans_labels2_pca = KMeans(n_clusters=4, random_state=42).fit_predict(x_pca2)
print('PCA + K-means silhouette score for B-cancer is :', silhouette_score(x_pca1, kmeans_labels1_pca, metric='euclidean'))
print('PCA + K-means silhouette score for Wifi is :', silhouette_score(x_pca2, kmeans_labels2_pca, metric='euclidean'))
print('PCA + K-means completeness score for B-cancer is :',  completeness_score(y1, kmeans_labels1_pca))
print('PCA + K-means completeness score for Wifi is :',  completeness_score(y2, kmeans_labels2_pca))

#Part3-1-2. KM + ICA
sse_kmeans1 = []
sse_kmeans2 = []
x_ica1 = FastICA(n_components=23, random_state=42, whiten='unit-variance').fit_transform(x1)
x_ica2 = FastICA(n_components=3, random_state=42, whiten='unit-variance').fit_transform(x2)
for i in range(2, N):
    kmeans1 = KMeans(n_clusters=i, random_state=42)
    kmeans1.fit(x_ica1)
    kmeans2 = KMeans(n_clusters=i, random_state=42)
    kmeans2.fit(x_ica2)
    sse_kmeans1.append(kmeans1.inertia_)
    sse_kmeans2.append(kmeans2.inertia_)
plt.figure(181)
plt.plot(np.arange(2,N,1),sse_kmeans1, label='B-cancer', marker='o',)
plt.title('Inertia of K-means after ICA: B-cancer')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid()
plt.savefig('fig18-1_kmeans_inertial_ica_breast.png')
plt.close()

plt.figure(182)
plt.plot(np.arange(2,N,1),sse_kmeans2, label='Wifi', marker='o',)
plt.title('Inertia of K-means after ICA: Wifi')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid()
plt.legend()
plt.savefig('fig18-2_kmeans_inertial_ica_wifi.png')
plt.close()

kmeans_labels1_ica = KMeans(n_clusters=10, random_state=42).fit_predict(x_ica1)
kmeans_labels2_ica = KMeans(n_clusters=4, random_state=42).fit_predict(x_ica2)
print('ICA + K-means silhouette score for B-cancer is :', silhouette_score(x_ica1, kmeans_labels1_ica, metric='euclidean'))
print('ICA + K-means silhouette score for Wifi is :', silhouette_score(x_ica2, kmeans_labels2_ica, metric='euclidean'))
print('ICA + K-means completeness score for B-cancer is :',  completeness_score(y1, kmeans_labels1_ica))
print('ICA + K-means completeness score for Wifi is :',  completeness_score(y2, kmeans_labels2_ica))

#Part3-1-3. KM + RP
sse_kmeans1 = []
sse_kmeans2 = []
x_rp1 = GaussianRandomProjection(n_components=5, random_state=42, compute_inverse_components=True).fit_transform(x1)
x_rp2 = GaussianRandomProjection(n_components=3, random_state=42, compute_inverse_components=True).fit_transform(x2)
for i in range(2, N):
    kmeans1 = KMeans(n_clusters=i, random_state=42)
    kmeans1.fit(x_rp1)
    kmeans2 = KMeans(n_clusters=i, random_state=42)
    kmeans2.fit(x_rp2)
    sse_kmeans1.append(kmeans1.inertia_)
    sse_kmeans2.append(kmeans2.inertia_)
plt.figure(191)
plt.plot(np.arange(2,N,1),sse_kmeans1, label='B-cancer', marker='o',)
plt.title('Inertia of K-means after RP: B-cancer')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid()
plt.savefig('fig19-1_kmeans_inertia_rp_breast.png')
plt.close()

plt.figure(192)
plt.plot(np.arange(2,N,1),sse_kmeans2, label='Wifi', marker='o',)
plt.title('Inertia of K-means after RP: Wifi')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid()
plt.savefig('fig19-2_kmeans_inertia_rp_wifi.png')
plt.close()


kmeans_labels1_rp = KMeans(n_clusters=7, random_state=42).fit_predict(x_rp1)
kmeans_labels2_rp = KMeans(n_clusters=4, random_state=42).fit_predict(x_rp2)
print('RP + K-means silhouette score for B-cancer is :', silhouette_score(x_rp1, kmeans_labels1_rp, metric='euclidean'))
print('RP + K-means silhouette score for Wifi is :', silhouette_score(x_rp2, kmeans_labels2_rp, metric='euclidean'))
print('RP + K-means completeness score for B-cancer is :',  completeness_score(y1, kmeans_labels1_rp))
print('RP + K-means completeness score for Wifi is :',  completeness_score(y2, kmeans_labels2_rp))

#Part3-1-4. KM B-cancer LLE
sse_kmeans1 = []
sse_kmeans2 = []
x_lle1 = LocallyLinearEmbedding(n_neighbors=11, n_components=17, random_state=42).fit_transform(x1)
x_lle2 = LocallyLinearEmbedding(n_neighbors=19, n_components=4, random_state=42).fit_transform(x2)
for i in range(2, N):
    kmeans1 = KMeans(n_clusters=i, random_state=42)
    kmeans1.fit(x_lle1)
    kmeans2 = KMeans(n_clusters=i, random_state=42)
    kmeans2.fit(x_lle2)
    sse_kmeans1.append(kmeans1.inertia_)
    sse_kmeans2.append(kmeans2.inertia_)
plt.figure(201)
plt.plot(np.arange(2,N,1),sse_kmeans1, label='B-cancer', marker='o',)
plt.title('Inertia of K-means after LLE: B-cancer')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid()
plt.savefig('fig20-1_kmeans_inertia_lle_breast.png')
plt.close()

plt.figure(202)
plt.plot(np.arange(2,N,1),sse_kmeans2, label='Wifi', marker='o',)
plt.title('Inertia of K-means after LLE: Wifi')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid()
plt.savefig('fig20-2_kmeans_inertia_lle_wifi.png')
plt.close()

kmeans_labels1_lle = KMeans(n_clusters=15, random_state=42).fit_predict(x_lle1)
kmeans_labels2_lle = KMeans(n_clusters=5, random_state=42).fit_predict(x_lle2)
print('LLE + K-means silhouette score for B-cancer is :', silhouette_score(x_lle1, kmeans_labels1_lle, metric='euclidean'))
print('LLE + K-means silhouette score for Wifi is :', silhouette_score(x_lle2, kmeans_labels2_lle, metric='euclidean'))
print('LLE + K-means completeness score for B-cancer is :',  completeness_score(y1, kmeans_labels1_lle))
print('LLE + K-means completeness score for Wifi is :',  completeness_score(y2, kmeans_labels2_lle))

#Part3-2-1. EM Wifi PCA
bic_em1 = []
bic_em2 = []
for i in range(2, N):
    em1 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em1.fit(x_pca1)
    em2 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em2.fit(x_pca2)
    bic_em1.append(em1.bic(x_pca1))
    bic_em2.append(em2.bic(x_pca2))
plt.figure(211)
plt.plot(np.arange(2,N,1),bic_em1, label='B-cancer', marker='o',)
plt.title('BIC score for EM after PCA: B-cancer')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig21-1_EM_Bic_score_pca_breast.png')
plt.close()

plt.figure(212)
plt.plot(np.arange(2,N,1),bic_em2, label='Wifi', marker='o',)
plt.title('BIC score for EM after PCA: Wifi')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig21-2_EM_Bic_score_pca_wifi.png')
plt.close()

em_labels1_pca = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit_predict(x_pca1)
em_labels2_pca = GaussianMixture(n_components=8, covariance_type='full', random_state=42).fit_predict(x_pca2)
print('PCA + EM silhouette score for B-cancer is :', silhouette_score(x_pca1, em_labels1_pca, metric='euclidean'))
print('PCA + EM silhouette score for Wifi is :', silhouette_score(x_pca2, em_labels2_pca, metric='euclidean'))
print('PCA + EM completeness score for B-cancer is :',  completeness_score(y1, em_labels1_pca))
print('PCA + EM completeness score for Wifi is :',  completeness_score(y2, em_labels2_pca))

#Part3-2-2. EM Wifi ICA
bic_em1 = []
bic_em2 = []
for i in range(2, N):
    em1 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em1.fit(x_ica1)
    em2 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em2.fit(x_ica2)
    bic_em1.append(em1.bic(x_ica1))
    bic_em2.append(em2.bic(x_ica2))
plt.figure(221)
plt.plot(np.arange(2,N,1),bic_em1, label='B-cancer', marker='o',)
plt.title('BIC score for EM after ICA: B-cancer')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig22-1_EM_Bic_score_ica_breast.png')
plt.close()

plt.figure(22)
plt.plot(np.arange(2,N,1),bic_em2, label='Wifi', marker='o',)
plt.title('BIC score for EM after ICA: Wifi')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig22-2_EM_Bic_score_ica_wifi.png')
plt.close()

em_labels1_ica = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit_predict(x_ica1)
em_labels2_ica = GaussianMixture(n_components=8, covariance_type='full', random_state=42).fit_predict(x_ica2)
print('ICA + EM silhouette score for B-cancer is :', silhouette_score(x_ica1, em_labels1_ica, metric='euclidean'))
print('ICA + EM silhouette score for Wifi is :', silhouette_score(x_ica2, em_labels2_ica, metric='euclidean'))
print('ICA + EM completeness score for B-cancer is :',  completeness_score(y1, em_labels1_ica))
print('ICA + EM completeness score for Wifi is :',  completeness_score(y2, em_labels2_ica))

#Part3-2-3. EM Wifi RP
bic_em1 = []
bic_em2 = []
for i in range(2, N):
    em1 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em1.fit(x_rp1)
    em2 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em2.fit(x_rp2)
    bic_em1.append(em1.bic(x_rp1))
    bic_em2.append(em2.bic(x_rp2))
plt.figure(231)
plt.plot(np.arange(2,N,1),bic_em1, label='B-cancer', marker='o',)
plt.title('BIC score for EM after RP: B-cancer')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig23-1_EM_Bic_score_rp_breast.png')
plt.close()

plt.figure(232)
plt.plot(np.arange(2,N,1),bic_em2, label='Wifi', marker='o',)
plt.title('BIC score for EM after RP: Wifi')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig23-2_EM_Bic_score_rp_wifi.png')
plt.close()

em_labels1_rp = GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit_predict(x_rp1)
em_labels2_rp = GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit_predict(x_rp2)
print('RP + EM silhouette score for B-cancer is :', silhouette_score(x_rp1, em_labels1_rp, metric='euclidean'))
print('RP + EM silhouette score for Wifi is :', silhouette_score(x_rp2, em_labels2_rp, metric='euclidean'))
print('RP + EM completeness score for B-cancer is :',  completeness_score(y1, em_labels1_rp))
print('RP + EM completeness score for Wifi is :',  completeness_score(y2, em_labels2_rp))

#Part3-2-4. EKM Wifi LLE
bic_em1 = []
bic_em2 = []
for i in range(2, N):
    em1 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em1.fit(x_lle1)
    em2 = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    em2.fit(x_lle2)
    bic_em1.append(em1.bic(x_lle1))
    bic_em2.append(em2.bic(x_lle2))
plt.figure(241)
plt.plot(np.arange(2,N,1),bic_em1, label='B-cancer', marker='o',)
plt.title('BIC score for EM after LLE: B-cancer')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig24-1_EM_Bic_score_lle_breast.png')
plt.close()

plt.figure(242)
plt.plot(np.arange(2,N,1),bic_em2, label='Wifi', marker='o',)
plt.title('BIC score for EM after LLE: Wifi')
plt.xlabel('number of components')
plt.ylabel('BIC score')
plt.grid()
plt.savefig('fig24-2_EM_Bic_score_lle_wifi.png')
plt.close()
em_labels1_lle = GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit_predict(x_lle1)
em_labels2_lle = GaussianMixture(n_components=8, covariance_type='full', random_state=42).fit_predict(x_lle2)
print('LLE + EM silhouette score for B-cancer is :', silhouette_score(x_lle1, em_labels1_lle, metric='euclidean'))
print('LLE + EM silhouette score for Wifi is :', silhouette_score(x_lle2, em_labels2_lle, metric='euclidean'))
print('LLE + EM completeness score for B-cancer is :',  completeness_score(y1, em_labels1_lle))
print('LLE + EM completeness score for Wifi is :',  completeness_score(y2, em_labels2_lle))

# Code reference from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#
plt.figure(251)
range_clusters=np.arange(7,8)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x_pca1) + (i + 1) * 10])
    km = KMeans(n_clusters=i, random_state=42)
    km_labels = km.fit_predict(x_pca1)
    sample_sil_values = silhouette_samples(x_pca1, km_labels)
    km_silhouette_avg = silhouette_score(x_pca1, km_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[km_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=km_silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(km_labels.astype(float) / i)
    ax2.scatter(x_pca1[:, 0], x_pca1[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = km.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle("Silhouette analysis for KM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig25-%d_silhouette_breast_km.png' %i)
    plt.close()

plt.figure(252)
range_clusters=np.arange(4,5)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x_pca2) + (i + 1) * 10])
    km = KMeans(n_clusters=i, random_state=42)
    km_labels = km.fit_predict(x_pca2)
    sample_sil_values = silhouette_samples(x_pca2, km_labels)
    km_silhouette_avg = silhouette_score(x_pca2, km_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[km_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=km_silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(km_labels.astype(float) / i)
    ax2.scatter(x_pca2[:, 0], x_pca2[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    # Labeling the clusters
    centers = km.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle("Silhouette analysis for KM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig25-%d_silhouette_wifi_km.png' %i)
    plt.close()

plt.figure(253)
range_clusters=np.arange(2,3)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x_pca1) + (i + 1) * 10])
    em = GaussianMixture(n_components=i, random_state=42)
    em_labels = em.fit_predict(x_pca1)
    sample_sil_values = silhouette_samples(x_pca1, em_labels)
    em_silhouette_avg = silhouette_score(x_pca1, em_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[em_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=em_silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(em_labels.astype(float) / i)
    ax2.scatter(x_pca1[:, 0], x_pca1[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    # Labeling the clusters
    centers = np.zeros((i, x_pca1.shape[1]))
    #centers = em.cluster_centers_
    for m in range(i):
        density = scipy.stats.multivariate_normal(cov=em.covariances_[m], mean=em.means_[m]).logpdf(x_pca1)
        centers[m, :] = x_pca1[np.argmax(density)]
        # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle("Silhouette analysis for EM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig25-%d_silhouette_breast_em.png' %i)
    plt.close()

plt.figure(254)
range_clusters=np.arange(8,9)
for i in range_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x_pca2) + (i + 1) * 10])
    em = GaussianMixture(n_components=i, random_state=42)
    em_labels = em.fit_predict(x_pca2)
    sample_sil_values = silhouette_samples(x_pca2, em_labels)
    em_silhouette_avg = silhouette_score(x_pca2, em_labels, metric='euclidean')
    y_lower=10
    for j in range(i):
        j_sil_values = sample_sil_values[em_labels==j]
        j_sil_values.sort()
        j_size = j_sil_values.shape[0]
        y_upper = y_lower + j_size
        color=cm.nipy_spectral(float(j)/i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, j_sil_values, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower+0.5*j_size, str(j))
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=em_silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(em_labels.astype(float) / i)
    ax2.scatter(x_pca2[:, 0], x_pca2[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    # Labeling the clusters
    centers = np.zeros((i, x_pca2.shape[1]))
    #centers = em.cluster_centers_
    for m in range(i):
        density = scipy.stats.multivariate_normal(cov=em.covariances_[m], mean=em.means_[m]).logpdf(x_pca2)
        centers[m, :] = x_pca2[np.argmax(density)]
        # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for k, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % k, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle("Silhouette analysis for EM on sample data with n_clusters = %d" %i,fontsize=14,fontweight="bold",)
    plt.savefig('fig25-%d_silhouette_wifi_em.png' %i)
    plt.close()

#Part4. Reduced data: For Wifi dataset
Xtrain_pca, Xtest_pca,Ytrain_pca, Ytest_pca = train_test_split(x_pca2, y2, test_size = 0.4, random_state=42, shuffle = True)
Xtrain_ica, Xtest_ica,Ytrain_ica, Ytest_ica = train_test_split(x_ica2, y2, test_size = 0.4, random_state=42, shuffle = True)
Xtrain_rp, Xtest_rp,Ytrain_rp, Ytest_rp = train_test_split(x_rp2, y2, test_size = 0.4, random_state=42, shuffle = True)
Xtrain_lle, Xtest_lle,Ytrain_lle, Ytest_lle = train_test_split(x_lle2, y2, test_size = 0.4, random_state=42, shuffle = True)
wall_time=[]
accuracy=[]

def NN_data(Xtrain, Xtest, Ytrain, Ytest):
    parameters = {'hidden_layer_sizes': np.arange(20, 45, 5), 'alpha': np.logspace(-3, 0, 8), 'learning_rate_init': np.logspace(-3, 0, 8)}
    clf = MLPClassifier(max_iter=2000, solver='adam', activation='relu', random_state=42, early_stopping=True, warm_start=True)
    best_classifier = GridSearchCV(clf, param_grid=parameters, error_score='raise',n_jobs=-1, scoring='accuracy', refit=True)
    start_time = time.time()
    best_classifier.fit(Xtrain, Ytrain)
    end_time = time.time()
    time_sec = end_time-start_time
    classifier_accuracy = accuracy_score(Ytest, best_classifier.predict(Xtest))
    print('best classifier for MLP: ', best_classifier.best_params_)
    return best_classifier, classifier_accuracy, time_sec

def plot_learning_curve_nn(Xtrain, Ytrain, best_classifier, cv):
    _, train_score, validation_score = learning_curve(best_classifier, Xtrain, Ytrain, train_sizes= np.linspace(0.1,1.0,10), cv = cv, n_jobs=-1)
    return train_score, validation_score

best_classifier_pca, accuracy_pca, time_pca=NN_data(Xtrain_pca, Xtest_pca,Ytrain_pca, Ytest_pca)
accuracy.append(accuracy_pca)
wall_time.append(time_pca)
best_classifier_ica, accuracy_ica, time_ica=NN_data(Xtrain_ica, Xtest_ica,Ytrain_ica, Ytest_ica)
accuracy.append(accuracy_ica)
wall_time.append(time_ica)
best_classifier_rp, accuracy_rp, time_rp=NN_data(Xtrain_rp, Xtest_rp,Ytrain_rp, Ytest_rp)
accuracy.append(accuracy_rp)
wall_time.append(time_rp)
best_classifier_lle, accuracy_lle, time_lle=NN_data(Xtrain_lle, Xtest_lle,Ytrain_lle, Ytest_lle)
accuracy.append(accuracy_lle)
wall_time.append(time_lle)

plt.figure(26)
compare = ['PCA', 'ICA', 'RP', 'LLE']
xx = np.arange(len(compare))
bar = plt.bar(xx, accuracy)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('Accuracy score of each dimension reductions')
plt.xticks(xx, compare)
plt.ylabel('accuracy(%)')
plt.tight_layout()
plt.savefig('fig26_accuracy_DR.png')
plt.close()

plt.figure(27)
compare = ['PCA', 'ICA', 'RP', 'LLE']
xx = np.arange(len(compare))
bar = plt.bar(xx, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('training time of each dimension reductions')
plt.xticks(xx, compare)
plt.ylabel('time(sec)')
plt.tight_layout()
plt.savefig('fig27_time_DR.png')
plt.close()

train_score_pca, validation_score_pca=plot_learning_curve_nn(Xtrain_pca, Ytrain_pca, best_classifier_pca, 10)
train_score_ica, validation_score_ica=plot_learning_curve_nn(Xtrain_ica, Ytrain_ica, best_classifier_ica, 10)
train_score_rp, validation_score_rp=plot_learning_curve_nn(Xtrain_rp, Ytrain_rp, best_classifier_rp, 10)
train_score_lle, validation_score_lle=plot_learning_curve_nn(Xtrain_lle, Ytrain_lle, best_classifier_lle, 10)

plt.figure(28)
plt.title("Learning Curve of NN after PCA")
plt.xlabel("samples")
plt.ylabel("Score")
plt.plot(np.mean(train_score_pca, axis=1), label='train score', marker='o', color='blue', lw=1.5)
plt.plot(np.mean(validation_score_pca, axis=1), label='valid score', marker='o',color='red', lw=1.5)
plt.legend()
plt.grid()
ind = np.arange(len(np.mean(validation_score_pca, axis=1)))
labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
plt.xticks(ind, labels)
plt.tight_layout()
plt.savefig('fig28_learning_curve_pca.png')
plt.close()

plt.figure(29)
plt.title("Learning Curve of NN after ICA")
plt.xlabel("samples")
plt.ylabel("Score")
plt.plot(np.mean(train_score_ica, axis=1), label='train score', marker='o', color='blue', lw=1.5)
plt.plot(np.mean(validation_score_ica, axis=1), label='valid score', marker='o',color='red', lw=1.5)
plt.legend()
plt.grid()
ind = np.arange(len(np.mean(validation_score_ica, axis=1)))
labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
plt.xticks(ind, labels)
plt.tight_layout()
plt.savefig('fig29_learning_curve_ica.png')
plt.close()

plt.figure(30)
plt.title("Learning Curve of NN after RP")
plt.xlabel("samples")
plt.ylabel("Score")
plt.plot(np.mean(train_score_rp, axis=1), label='train score', marker='o', color='blue', lw=1.5)
plt.plot(np.mean(validation_score_rp, axis=1), label='valid score', marker='o',color='red', lw=1.5)
plt.legend()
plt.grid()
ind = np.arange(len(np.mean(validation_score_rp, axis=1)))
labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
plt.xticks(ind, labels)
plt.tight_layout()
plt.savefig('fig30_learning_curve_rp.png')
plt.close()

plt.figure(31)
plt.title("Learning Curve of NN after LLE")
plt.xlabel("samples")
plt.ylabel("Score")
plt.plot(np.mean(train_score_lle, axis=1), label='train score', marker='o', color='blue', lw=1.5)
plt.plot(np.mean(validation_score_lle, axis=1), label='valid score', marker='o',color='red', lw=1.5)
plt.legend()
plt.grid()
ind = np.arange(len(np.mean(validation_score_lle, axis=1)))
labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
plt.xticks(ind, labels)
plt.tight_layout()
plt.savefig('fig31_learning_curve_lle.png')
plt.close()


def plot_loss(Xtrain, Ytrain, best_classifier, epoch):
    loss = np.zeros(epoch)
    clf = MLPClassifier(max_iter=1, solver='adam', activation='relu', random_state=42, warm_start=True, alpha=best_classifier.best_params_['alpha'],hidden_layer_sizes=best_classifier.best_params_['hidden_layer_sizes'],learning_rate_init=best_classifier.best_params_['learning_rate_init'])
    for i in range(epoch):
        clf.fit(Xtrain, Ytrain)
        loss[i] = log_loss(Ytrain, clf.predict_proba(Xtrain))
    return loss

epoch=500
plt.figure(32)
loss=plot_loss(Xtrain_pca, Ytrain_pca, best_classifier_pca, epoch)
plt.plot(np.arange(epoch), loss)
plt.title("Loss function of NN after PCA")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('fig32_pca_loss.png')
plt.close()

plt.figure(33)
loss=plot_loss(Xtrain_ica, Ytrain_ica, best_classifier_ica, epoch)
plt.plot(np.arange(epoch), loss)
plt.title("Loss function of NN after ICA")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('fig33_ica_loss.png')
plt.close()

plt.figure(34)
loss=plot_loss(Xtrain_rp, Ytrain_rp, best_classifier_rp, epoch)
plt.plot(np.arange(epoch), loss)
plt.title("Loss function of NN after RP")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('fig34_rp_loss.png')
plt.close()

plt.figure(35)
loss=plot_loss(Xtrain_lle, Ytrain_lle, best_classifier_lle, epoch)
plt.plot(np.arange(epoch), loss)
plt.title("Loss function of NN after LLE")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('fig35_lle_loss.png')
plt.close()

#Part5. NN + Clustering for Wifi dataset
wall_time=[]
accuracy=[]
x_extend_km=np.zeros((x2.shape[0], x2.shape[1]+1))
x_extend_km[:,:-1]=x2
x_extend_km[:,-1]=kmeans_labels2
Xtrain_km, Xtest_km, Ytrain_km, Ytest_km=train_test_split(x_extend_km, y2, train_size=0.4, random_state=42, shuffle=True)

x_extend_em=np.zeros((x2.shape[0], x2.shape[1]+1))
x_extend_em[:,:-1]=x2
x_extend_em[:,-1]=em_labels2
Xtrain_em, Xtest_em, Ytrain_em, Ytest_em=train_test_split(x_extend_em, y2, train_size=0.4, random_state=42, shuffle=True)

best_classifier_km, accuracy_km, time_km=NN_data(Xtrain_km, Xtest_km, Ytrain_km, Ytest_km)
accuracy.append(accuracy_km)
wall_time.append(time_km)
best_classifier_em, accuracy_em, time_em=NN_data(Xtrain_em, Xtest_em, Ytrain_em, Ytest_em)
accuracy.append(accuracy_em)
wall_time.append(time_em)

plt.figure(36)
compare = ['KM', 'EM']
xx = np.arange(len(compare))
bar = plt.bar(xx, accuracy)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('Accuracy score of each clustering')
plt.xticks(xx, compare)
plt.ylabel('accuracy(%)')
plt.tight_layout()
plt.savefig('fig36_accuracy_.png')
plt.close()

plt.figure(37)
compare = ['KM', 'EM']
xx = np.arange(len(compare))
bar = plt.bar(xx, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('training time of each clustering')
plt.xticks(xx, compare)
plt.ylabel('time(sec)')
plt.tight_layout()
plt.savefig('fig37_time_.png')
plt.close()

train_score_km, validation_score_km=plot_learning_curve_nn(Xtrain_km, Ytrain_km, best_classifier_km, 10)
train_score_em, validation_score_em=plot_learning_curve_nn(Xtrain_em, Ytrain_em, best_classifier_em, 10)

plt.figure(38)
plt.title("Learning Curve of NN after KM")
plt.xlabel("samples")
plt.ylabel("Score")
plt.plot(np.mean(train_score_km, axis=1), label='train score', marker='o', color='blue', lw=1.5)
plt.plot(np.mean(validation_score_km, axis=1), label='valid score', marker='o',color='red', lw=1.5)
plt.legend()
plt.grid()
ind = np.arange(len(np.mean(validation_score_km, axis=1)))
labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
plt.xticks(ind, labels)
plt.tight_layout()
plt.savefig('fig38_learning_curve_km.png')
plt.close()


plt.figure(39)
plt.title("Learning Curve of NN after EM")
plt.xlabel("samples")
plt.ylabel("Score")
plt.plot(np.mean(train_score_em, axis=1), label='train score', marker='o', color='blue', lw=1.5)
plt.plot(np.mean(validation_score_em, axis=1), label='valid score', marker='o',color='red', lw=1.5)
plt.legend()
plt.grid()
ind = np.arange(len(np.mean(validation_score_em, axis=1)))
labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
plt.xticks(ind, labels)
plt.tight_layout()
plt.savefig('fig39_learning_curve_em.png')
plt.close()

plt.figure(40)
loss=plot_loss(Xtrain_km, Ytrain_km, best_classifier_km, epoch)
plt.plot(np.arange(epoch), loss)
plt.title("Loss function of NN after KM")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('fig40_km_loss.png')
plt.close()

plt.figure(41)
loss=plot_loss(Xtrain_em, Ytrain_em, best_classifier_em, epoch)
plt.plot(np.arange(epoch), loss)
plt.title("Loss function of NN after EM")
plt.xlabel("Number of epochs")
plt.ylabel("Loss value")
plt.grid()
plt.tight_layout()
plt.savefig('fig41_em_loss.png')
plt.close()