from fitsio import FITS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from os import scandir
from datetime import datetime
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse
import logging


logger = logging.getLogger('AGILE_classifier')

try:
    import colorlog

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s %(levelname)s: %(message)s'))
    logger.addHandler(handler)
except ImportError:
    # file_handler = logging.FileHandler(filename='AGILE.log')
    stdout_handler = logging.StreamHandler()
    # handlers = [file_handler, stdout_handler]
    logging.basicConfig(format="%(levelname)s: %(message)s", handlers=stdout_handler)

logger.setLevel(logging.INFO)


def save_model(model, modelfilename):
    """Save trained classificator to a pickle file in order to be loaded for successive predictions"""
    dump_dict = {'model':model}

    try:
        _ = joblib.dump(dump_dict, modelfilename + '.pkl', compress=3)
    except IOError:
        raise IOError('Could not save model info to disk, please check '
                      'dir writing permissions and space left on disk')


def read_data(path):
    """Read input data with *.flg.gz extension from input path, which has to be a directory.
       Returns a pandas Series for EVTSTATUS, and a pandas DataFrame of features"""

    col2del = ['BRNAME01', 'BRNAME02', 'BRNAME03', 'EVSTATUS']
    datalist = list()
    statuslist = list()
    for f in scandir(path):
        if f.name.endswith('.flg.gz'):
            with FITS(path + '/' + f.name) as hdu:
                flgdata = hdu['FLAGS'].read()

                # create label separately and then remove from df outside this function
                evstatus = flgdata['EVSTATUS'].astype(str)
                # about byteswap see: https://stackoverflow.com/a/30284033/827818
                st_df = pd.Series(evstatus.byteswap().newbyteorder())
                statuslist.append(st_df)

                # features
                flg_df = pd.DataFrame(flgdata.byteswap().newbyteorder())
                flg_df.drop(col2del, inplace=True, axis=1)
                datalist.append(flg_df)

    statuses = pd.concat(statuslist, ignore_index=True)
    feature_df = pd.concat(datalist, ignore_index=True)

    return statuses, feature_df


def plot_confusion_matrix(cm, classes, model,
                          normalize=False,
                          title='Confusion_matrix',
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

    logger.info("Confusion matrix: \n{}".format(cm))

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


def dump2csv(timecol, ytrue, ypred, model):
    """Dumps results to a csv file with ' ' separator, with header.
       The dumped file has three columns:
       - TIME column from the original flagged file, meant to be the event ID
       - KALMAN_FLAG, which is the original label
       - PREDICTED_FLAG, which is this classificator prediction"""
    dum = pd.DataFrame({'TIME':timecol, 'TRUE_LABEL':ytrue, 'PRED_LABEL':ypred})
    outfile = datetime.now().strftime('%Y%m%d_%H.%M_') + model + '_results.csv'
    dum.to_csv(outfile, sep=' ', index=False, header=['TIME', 'KALMAN_FLAG', 'PREDICTED_FLAG'])


def parse_args():
    # Declare and parse command line option
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', dest='input_dir', required=True, help='input directory')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # read the data into two pandas dataframes
    evstatus, flg_df = read_data(args.input_dir)

    # evstatus = pd.concat(evstatus_list, ignore_index=True)
    # flg_df = pd.concat(flg_list, ignore_index=True)

    # Encode labels to ints for processing
    le = LabelEncoder()
    labels = le.fit_transform(evstatus.values)
    classes_names = le.classes_

    logger.info("Classes: {}".format(classes_names))

    X = flg_df.values

    # separate the input features to perform both training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.75,
                                                        random_state=21, stratify=labels)

    # scale the features with mean and std
    # see http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # RF
    clf = RandomForestClassifier(n_estimators=50, max_features="sqrt", n_jobs=2,
                                 oob_score=True, random_state=42, criterion='entropy')
    clf.fit(X_train_scaled, y_train)
    logger.info("Feature importances: {}".format(clf.feature_importances_))
    save_model(clf, 'random_forest')
    y_pred = clf.predict(X_test_scaled)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    dump2csv(X_test[:, 0], le.inverse_transform(y_test), le.inverse_transform(y_pred), 'RF')

    plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=classes_names, model='RF', normalize=True)
    plot_confusion_matrix(cnf_matrix, classes=classes_names, model='RF')

    # ---------------------------- #

    # NN
    nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(40,), random_state=42)
    nn.fit(X_train_scaled, y_train)
    save_model(nn, 'neural_network')
    nn_y_pred = nn.predict(X_test_scaled)

    nn_cnf_matrix = confusion_matrix(y_test, nn_y_pred)

    dump2csv(X_test[:, 0], le.inverse_transform(y_test), le.inverse_transform(nn_y_pred), 'NN')

    plt.figure()
    # plot_confusion_matrix(nn_cnf_matrix, classes=classes_names, model='NN', normalize=True)
    plot_confusion_matrix(nn_cnf_matrix, classes=classes_names, model='NN')


if __name__ == '__main__':
    main()