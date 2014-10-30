# SVM results
from sklearn.svm import SVC
from sklearn import metrics


fig = plt.figure(figsize=(6, 6))

i = 0
for kernel in ['rbf', 'linear']:
    clf = SVC(kernel=kernel).fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    label = "SVC: kernel = {0}".format(kernel)
    print('{0}: {1}'.format(label, metrics.f1_score(ytest, ypred))
    ax = fig.add_subplot(3, 2, i + 1, xticks=[], yticks=[])
    ax.imshow(metrics.confusion_matrix(ypred, ytest),
               interpolation='nearest', cmap=plt.cm.binary)
    #ax.colorbar()
    ax.set_xlabel("true label")
    ax.set_ylabel("predicted label")
    ax.set_title(label)
    i += 1

# random forest results
from sklearn.ensemble import RandomForestClassifier

for max_features in [1.0, 0.3]:
    for n_estimators in [10, 100]:
        clf = RandomForestClassifier(max_features=max_features,
                                     n_estimators=n_estimators,
                                     random_state=0).fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        label = "RF: max_features = {0}, n_estimators = {1}".format(
            max_features, n_estimators)
        print('{0}: {1}'.format(label, metrics.f1_score(ytest, ypred))
        ax = fig.add_subplot(3, 2, i + 1, xticks=[], yticks=[])
        ax.imshow(metrics.confusion_matrix(ypred, ytest),
                   interpolation='nearest', cmap=plt.cm.binary)
        #ax.colorbar()
        ax.set_xlabel("true label")
        ax.set_ylabel("predicted label")
        ax.set_title(label)
        i += 1
