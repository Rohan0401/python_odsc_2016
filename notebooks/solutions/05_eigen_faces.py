from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

faces = fetch_olivetti_faces()

X = faces.data
pca = PCA(n_components=100).fit(X)

# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

eigen_faces = pca.components_.reshape((pca.components_.shape[0], 64, 64))

# plot the eigen faces:
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(eigen_faces[i], cmap=plt.cm.bone, interpolation='nearest')


# plot the reconstruction

# compute embedding
X_emb = pca.transform(X)

# compute reconstruction
X_rec = np.dot(X_emb, pca.components_).reshape((X_emb.shape[0], 64, 64))

# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the faces:
for i in range(0, 16, 2):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone, interpolation='nearest')
    ax = fig.add_subplot(4, 4, i + 2, xticks=[], yticks=[])
    ax.imshow(X_rec[i], cmap=plt.cm.bone, interpolation='nearest')
