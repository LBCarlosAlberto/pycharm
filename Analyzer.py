from sklearn.model_selection import train_test_split
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy import io
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn import metrics
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
#classifiers choose from 'nb', 'svm', 'rf'
def grid_search_classifier(x_train, y_train, sample_size = 1000,\
 replace = True,classifier = 'svm', parameters = {}, cv = 5, metric = 'f1_macro'):
    if classifier == 'svm':
        clf = svm.SVC(decision_function_shape='ovr')
        if len(parameters) == 0:
            parameters = {'C': [1, 5, 10]}
    elif classifier == 'nb':
        clf = MultinomialNB()
        if len(parameters) == 0:
            parameters = {'alpha': [0.1, 0.5, 1, 2]}
    else:
        clf = RandomForestClassifier(n_estimators = 1000, random_state = 8)
        if len(parameters) == 0:
            parameters = {'max_feature': ('auto', 'sqrt'), 'class_weight':('balanced', None, 'balanced_subsample')}
    x, y = resample(x_train, y_train,\
        n_samples = sample_size, random_state = 20, replace = replace)
    gs_clf = GridSearchCV(clf, param_grid=parameters, scoring=metric, cv = 5)
    gs_clf = gs_clf.fit(x, y)
    best_model, best_parameters = gs_clf.best_estimator_, gs_clf.best_params_
    return best_model, best_parameters

def train_model(classifier, X_train, Y_train, sample_size = 1000):
    #load in data
    #2 grams features

    if classifier == 'nb':
        best_model, best_params = grid_search_classifier(X_train, Y_train, \
                    sample_size = sample_size, replace = False, classifier = classifier,\
                     parameters = {}, cv = 5, metric = 'f1_macro')
    elif classifier == 'svm':
        svc_parameters = {'kernel' : ['linear'],
                 'C': [0.2, 1, 10, 50, 100],
                  'class_weight':[None, 'balanced'],
                  'gamma': [0.001, 0.01, 0.1, 1]
                  }
        best_model, best_params = grid_search_classifier(X_train, Y_train, \
                    sample_size = sample_size, replace = False, classifier = classifier,\
                     parameters = svc_parameters, cv = 5, metric = 'f1_macro')
    else:
        rb_params = {"max_depth": [3, None, 5, 10],
              "max_features": ['auto', 'sqrt', 'log2', 100],
              "min_samples_split": [10, 20],
              "min_samples_leaf": [3, 10, 20],
              'class_weight':['balanced', None, 'balanced_subsample'],
              "criterion": ["gini", "entropy"]}
        best_model, best_params = grid_search_classifier(X_train, Y_train, \
                    sample_size = sample_size, replace = False, classifier = classifier,\
                     parameters = rb_params, cv = 5, metric = 'f1_macro')
    best_model.fit(X_train, Y_train)
    return best_model, best_params

def evaluate_and_report(trained_model, x_test, y_test, tag_names):
    predicted = trained_model.predict(x_test)
    print(classification_report(y_test, predicted, target_names=tag_names))
    return predicted

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
if __name__ == '__main__':
    model = 'svm'
    print "model:", model
    topics = ['business','environment','fashion','lifeandstyle',\
                'politics','sport','technology','travel','world']
    train = io.mmread('1gram_train.mtx')
    test = io.mmread('1gram_test.mtx')
    train_tags = pickle.load(open('1gram_train_tags.pk', 'r'))
    test_tags = pickle.load(open('1gram_test_tags.pk', 'r'))
    best_model, best_params = train_model(model, train, train_tags)
    model_file = 'train_model_' + model + '.sav'
    pickle.dump(best_model, open(model_file, 'wb'))
    prediction = evaluate_and_report(best_model, test, test_tags, topics)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_tags, prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(cnf_matrix, classes=topics,
                      title='Confusion matrix, without normalization')
    #plt.show()
    image_name = 'confusion_matrix_' + 'model' + '.png'
    savefig(image_name)
    
