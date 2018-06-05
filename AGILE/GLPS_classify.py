from fitsio import FITS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, model,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(model + '_' + title + '.png')

    if show:
        plt.show()


def main():
    hdu = FITS('data/PKP057558_1_3901_000_1527734767.flg.gz')
    flg = hdu['FLAGS'].read()
    evstatus = flg['EVSTATUS'].astype(str)

    le = LabelEncoder()
    labels = le.fit_transform(evstatus)
    classes_names = le.classes_

    flg_df = pd.DataFrame(flg.byteswap().newbyteorder())

    col2del = ['BRNAME01', 'BRNAME02', 'BRNAME03', 'EVSTATUS']
    flg_df.drop(col2del, inplace=True, axis=1)

    X = flg_df.values

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=21)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    clf = RandomForestClassifier(n_estimators=50, max_features="sqrt", n_jobs=2,
                                 oob_score=True, random_state=42, criterion='entropy')

    nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50,), random_state=42)


    # RF
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=classes_names,
    #                       title='Confusion matrix, without normalization')

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes_names, model='RF', normalize=True)

    # NN
    nn.fit(X_train, y_train)
    nn_y_pred = nn.predict(X_test)

    nn_cnf_matrix = confusion_matrix(y_test, nn_y_pred)
    # plt.figure()
    # plot_confusion_matrix(nn_cnf_matrix, classes=classes_names,
    #                       title='NN Confusion matrix, without normalization')

    plt.figure()
    plot_confusion_matrix(nn_cnf_matrix, classes=classes_names, model='NN', normalize=True)


if __name__ == '__main__':
    main()