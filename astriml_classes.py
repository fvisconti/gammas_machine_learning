import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import fitsio
from astropy.io import fits
from os import scandir
import logging
import time
from sklearn.externals import joblib


logger = logging.getLogger('astriluts')


def get_true_energy(evtnum, mcrunnum, simtable):
    """Match true energy between evtable and simtable
    Returns an array of energies ordered like events in evtable"""

    ntrig = evtnum.shape[0]  # evtable["EVTNUM"].size
    nsim = simtable["MCEVTNUM"].size
    evt_mc = np.empty((ntrig, 2), dtype=np.int32)
    evt_mc[:, 0] = evtnum  # evtable["EVTNUM"]
    evt_mc[:, 1] = mcrunnum  # evtable["MCRUNNUM"]
    sim_mc = np.empty((nsim, 2), dtype=np.int32)
    sim_mc[:, 0] = simtable["MCEVTNUM"]
    sim_mc[:, 1] = simtable["MCRUNNUM"]
    e0 = simtable['E0']

    idx = np.empty(ntrig, dtype=np.int32)

    for i, row in enumerate(evt_mc):
        idx[i] = np.where((sim_mc[:, 0] == row[0]) * (sim_mc[:, 1] == row[1]))[0]

    energy = e0[idx]

    return energy


class EventsRecoTrainer(object):

    def __init__(self, params):
        self.read_common_parameters(params)

        if self.do_gh:
            self.read_gh_sep_parameters(params)
        if self.do_en:
            self.read_enreco_parameters(params)
        if self.do_dir:
            self.read_dirreco_parameters(params)

    def read_common_parameters(self, params):
        try:
            self.gamma_dir = params['gamma_dir']
            self.proton_dir = params['proton_dir']
            self.evtable_name = params['evtable']
            self.file_ext = params['file_ext']
            self.do_gh = params['do_gh_sep']
            self.do_en = params['do_en_reco']
            self.do_dir = params['do_dir_reco']
            self.used_preprocessed_data = params['used_preprocessed_data']
        except IOError:
            raise IOError("Can't read commons in parfile, please check")

    def read_gh_sep_parameters(self, params):
        try:
            self.gh_filters = params['gh_filter_cut']
            self.gh_model_name = params['gh_model_name_sk']

            # read features to train on
            self.gh_num_features = params['gh_sk_num_par']

            self.gh_features = list()

            # then you can call feature_extractor with *features as args
            for i in range(self.gh_num_features):
                self.gh_features.append(params['gh_sk_par' + str(i + 1)])

            # read params for rebinning
            """build list of dictionaries like this:
            >>> featsToGroupBy = [{'feat': 'feat0', 'maxf': max(x[:, 0]), 'minf': min(x[:, 0]), 'col': 0, 'nbins': 10},
                ... {'feat': 'feat1', 'maxf': max(x[:, 1]), 'minf': min(x[:, 1]), 'col': 1, 'nbins': 5}]
            """
            self.gh_sep_params = [{'maxf': params['hi_resz_sk'], 'minf': params['lo_resz_sk'],
                                   'col': params['size_pos_sk'] - 1, 'nbins': params['n_resz_sk']},
                                  {'maxf': params['hi_redist0_sk'], 'minf': params['lo_redist0_sk'],
                                   'col': params['dist_pos_sk'] - 1, 'nbins': params['n_redist0_sk']},
                                  {'maxf': 1., 'minf': 0,  # for zenith rebinning it's fixed
                                   'col': params['zenith_pos_sk'] - 1, 'nbins': params['n_rezd_sk']},
                                  {'maxf': params['hi_reaz_sk'], 'minf': params['lo_reaz_sk'],
                                   'col': params['azimuth_pos_sk'] - 1, 'nbins': params['n_reaz_sk']},
                                 ]

            self.gh_rf_trees = params['gh_n_tree_sk']
            self.gh_rf_max_features = params['gh_max_features_sk']
            self.gh_rf_jobs = params['gh_n_jobs_sk']
            self.gh_rf_oob_score = params['gh_oob_ratio_sk']
            self.gh_rf_min_sample_split = params['gh_sample_split_sk']

        except IOError:
            raise IOError("Can't read gh separation stuff in parfile, please check")

    def read_enreco_parameters(self, params):

        try:
            self.en_filters = params['en_filter_cut']
            self.en_model_name = params['en_model_name_sk']

            # read features to train on
            self.en_num_features = params['en_sk_num_par']

            self.en_features = list()

            # then you can call feature_extractor with *features as args
            for i in range(self.en_num_features):
                self.en_features.append(params['en_sk_par' + str(i + 1)])

            self.en_rf_trees = params['en_n_tree_sk']
            self.en_rf_max_features = params['en_max_features_sk']
            self.en_rf_jobs = params['en_n_jobs_sk']
            self.en_rf_oob_score = params['en_oob_ratio_sk']

        except IOError:
            raise IOError("Can't read energy stuff in parfile, please check")

    def read_dirreco_parameters(self, params):

        try:
            self.dir_filters = params['dir_filter_cut']
            self.dir_model_name = params['dir_model_name_sk']

            # read features to train on
            self.dir_num_features = params['dir_sk_num_par']

            self.dir_features = list()

            # then you can call feature_extractor with *features as args
            for i in range(self.dir_num_features):
                self.dir_features.append(params['dir_sk_par' + str(i + 1)])

            self.dir_rf_trees = params['dir_n_tree_sk']
            self.dir_rf_max_features = params['dir_max_features_sk']
            self.dir_rf_jobs = params['dir_n_jobs_sk']
            self.dir_rf_oob_score = params['dir_oob_ratio_sk']

        except IOError:
            raise IOError("Can't read direction stuff in parfile, please check")

    @staticmethod
    def save_rf_model(model, filters, features_names, outfile):

        dump_dict = {'model':model, 'filters':filters, 'feature_names':features_names}

        try:
            _ = joblib.dump(dump_dict, outfile + '.pkl', compress=3)
        except IOError:
            raise IOError('Could not save model info to disk, please check '
                          'dir writing permissions and space left on disk')


    @staticmethod
    def features_extractor(path, filters, evtable, file_ext, npar, reco_type='gh', *args, **kwargs):
        """Generates event list with npar number of features defined in
        *args. To see what args can be, refer to astriluts_scikit.par
        Since it yields objects, it has to be called with np.stack (and squeeze)
        """
        # this imports allowed math functions
        from numpy import log10, cos, sin, abs

        evts = list()
        true_energy = list()
        dist = list()

        for f in scandir(path):
            if f.name.endswith(file_ext):
                with fitsio.FITS(path + f.name) as hdu:

                    mask = hdu[evtable].where(filters)

                    # NEVER EVER CHANGE events VARIABLE NAME!!!
                    # the events variable defined here is used in eval function
                    # see astriluts.par
                    events = hdu[evtable][mask]

                    evts_feat = np.zeros(shape=(len(events), npar))

                    # evts_feat = combine_features(evts).swapaxes(0, 1)
                    for i in range(npar):
                        # par_name = 'gh_par' + str(i)
                        evts_feat[:, i] = eval(args[i])

                    if reco_type == 'gh':
                        # put cos(ZD) and AZ columns at the end of evts_feat
                        evts_feat = np.concatenate((evts_feat, np.cos(events["ZD"].reshape(-1, 1))), axis=1)
                        evts_feat = np.concatenate((evts_feat, events["AZ"].reshape(-1, 1)), axis=1)

                    elif reco_type == 'en':
                        sim_evt = hdu["SIM_EVTS"].read()
                        temp_true_energy = get_true_energy(events["EVTNUM"], events["MCRUNNUM"], sim_evt)
                        true_energy.append(temp_true_energy)
                        # yield evts_feat, true_energy

                    elif reco_type == 'dir':
                        temp_dist = events["DIST"]
                        dist.append(temp_dist)


                    evts.append(evts_feat)
                        # yield evts_feat, dist

                    # # call this with e = (np.stack(features_extractor))
                    # else:
                    #     yield evts_feat

        if reco_type == 'gh':
            return np.vstack(evts)
        elif reco_type == 'en':
            return np.vstack(evts), np.hstack(true_energy)
        elif reco_type == 'dir':
            return np.vstack(evts), np.hstack(dist)

    def _hyperBinning(self, x, featsToGroupBy: list):
        """
        This function is for hyper binning with pandas.
        It is intended to be used here in order to level number of events before training the classifier;
        for more general purposes, it is the Histogram in utils/fitshistogram.py to be used.

        This outputs the input array grouped in as many bins as present in the featsToGroupBy list.
        This is a list of dictionaries, each dictionary is related to a feature (array column) to be binned.

        Beware of bins generation: since np.linspace is used, if you want log bins
        you have to make log of the input array column and pass log(max and min)!

        Parameters
        ----------
        x: input array to be binned

        featsToGroupBy: list(dictionary)
              list of dictionaries (see Examples for dict keys)

        Output
        ----------
        A pandas.core.groupby.DataFrameGroupBy object


        Examples
        --------

        >>> featsToGroupBy = [{'feat': 'feat0', 'maxf': max(x[:, 0]), 'minf': min(x[:, 0]), 'col': 0, 'nbins': 10},
        ... {'feat': 'feat1', 'maxf': max(x[:, 1]), 'minf': min(x[:, 1]), 'col': 1, 'nbins': 5}]

        where

        'feat' is feature name (optional);
        'maxf' and 'minf' are the range where to bin in
        'col' is feature column number in the input array
        'nbins' is the number of requested bins
        """
        dfx = pd.DataFrame(x)
        binning_list = list()

        for i in range(len(featsToGroupBy)):
            feat_dict = featsToGroupBy[i]
            bins = np.linspace(feat_dict['minf'], feat_dict['maxf'], feat_dict['nbins']+1)
            binning_list.append(pd.cut(dfx[feat_dict['col']], bins))

        groups = dfx.groupby(binning_list)

        return groups

    @staticmethod
    def _level_populations(group_signal, group_bgd, signal_evts, bgd_evts):
        """Equalize number of entries in each bin.
        When doing signal - background separation, it is common to wrangle input data equalizing
        the number of entries for gammas and hadron in all the requested hyper-bins,
        before training the classifier.
        This is different from the sklearn train_test_split in the sense that this level
        the two populations based upon a previous binning over any feature requested.

        Parameters
        ----------
        group_signal: (multi dimension) histogram of signal population in pandas groups format

        group_bgd: (multi dimension) histogram of background population in pandas groups format

        signal_evts: array of signal events, to be equalized

        bgd_evts: array of background events, to be equalized


        Output
        -----------
        array of gammas and array of hadrons now of the same size

        """

        # convert to pandas to use .drop()
        # dataframe names follow the original use case gamma (dfg) vs hadrons (dfh)
        dfg = pd.DataFrame(signal_evts)
        dfh = pd.DataFrame(bgd_evts)

        # have a unique set of keys in the histograms
        s = set(group_signal.indices)
        s.update(group_bgd.indices)

        for key in s:
            if key in group_signal.indices and key in group_bgd.indices:
                # count exceeding records
                exceeding = len(group_bgd.indices[key]) - len(group_signal.indices[key])

                # drop records from dataset picking exceeding number of indices
                # among those in group.indices[key]
                if exceeding > 0:
                    logger.debug('bin: {} - g: {}\th: {}\t - '
                                 'Removing {} protons'.format(key, len(group_signal.indices[key]),
                                                                   len(group_bgd.indices[key]),
                                                                   np.abs(exceeding)))
                    r_ind_list = list(np.random.choice(group_bgd.indices[key], size=exceeding, replace=False))
                    dfh.drop(r_ind_list, inplace=True)
                elif exceeding < 0:
                    logger.debug('bin: {} - g: {}\th: {}\t - '
                                 'Removing {} gammas'.format(key, len(group_signal.indices[key]),
                                                                  len(group_bgd.indices[key]),
                                                                  np.abs(exceeding)))
                    r_ind_list = list(np.random.choice(group_signal.indices[key], size=-exceeding, replace=False))
                    dfg.drop(r_ind_list, inplace=True)

            elif key in group_bgd.indices and key not in group_signal.indices:
                dfh.drop(group_bgd.indices[key], inplace=True)

            elif key in group_signal.indices and key not in group_bgd.indices:
                dfg.drop(group_signal.indices[key], inplace=True)

        return np.array(dfg), np.array(dfh)

    def setup_for_training(self, reco_type='gh', save=False):
        if reco_type == 'gh':
            gfeat = self.features_extractor(self.gamma_dir, self.gh_filters, self.evtable_name, self.file_ext,
                                            self.gh_num_features, reco_type, *self.gh_features)
            pfeat = self.features_extractor(self.proton_dir, self.gh_filters, self.evtable_name, self.file_ext,
                                            self.gh_num_features, reco_type, *self.gh_features)

            logger.info('Gamma events read: {}\nProtons events read: {}'.format(len(gfeat), len(pfeat)))

            groupg = self._hyperBinning(gfeat, self.gh_sep_params)
            groupp = self._hyperBinning(pfeat, self.gh_sep_params)

            signal, bgd = self._level_populations(groupg, groupp, gfeat, pfeat)

            num_gammas, num_protons = len(signal), len(bgd)

            logger.info('Gamma events after filtering: {}\n'
                        'Proton events after filtering: {}'.format(num_gammas, num_protons))

            # DEBUG: check for equal size
            assert (num_gammas == num_protons)

            # merge gamma and proton features
            self.features = np.concatenate([signal, bgd])

            # remove azimuth and zenith columns from features to train on
            self.features = np.delete(self.features, np.s_[-2:], axis=1)

            # merge labels
            label_g = np.ones(num_gammas)
            label_p = np.zeros(num_protons)
            self.target = np.concatenate([label_g, label_p])

            # save features and labels to disk
            if save:
                np.savez_compressed("features_and_labels_for_gh_training.npz", f=self.features, t=self.target)

        elif reco_type == 'en':
            self.features, self.target = self.features_extractor(self.gamma_dir, self.en_filters,
                                                                 self.evtable_name,
                                                                 self.file_ext, self.en_num_features, reco_type,
                                                                 *self.en_features)

            # self.features = np.vstack(temp_list[0])
            # self.target = np.concatenate(temp_list[1])

            if save:
                np.savez_compressed("gammas_for_enreco_training.npz", f=self.features, t=self.target)


        elif reco_type == 'dir':
            self.features, self.target = self.features_extractor(self.gamma_dir, self.dir_filters,
                                                                 self.evtable_name,
                                                                 self.file_ext, self.dir_num_features, reco_type,
                                                                 *self.dir_features)

            # self.features = np.vstack(temp_list[0])
            # self.target= np.hstack(temp_list[1])

            if save:
                np.savez_compressed("gammas_for_dirreco_training.npz", f=self.features, t=self.target)


    def train(self, reco_type='gh'):
        # self.setup_for_training(reco_type)

        if reco_type == 'gh':
            if self.used_preprocessed_data:
                with np.load("features_and_labels_for_gh_training.npz") as data:
                    self.features = data['f']
                    self.target = data['t']

            self.model = RandomForestClassifier(n_estimators=self.gh_rf_trees, max_features=self.gh_rf_max_features,
                                              n_jobs=self.gh_rf_jobs, oob_score=self.gh_rf_oob_score,
                                              random_state=42, criterion='entropy',
                                              min_samples_split=self.gh_rf_min_sample_split)

            logger.info('Start training RF classifier')
            start = time.process_time()
            self.model.fit(self.features, self.target)
            logger.info('...done. Training RF Classifier took {:.2f} s'.format(time.process_time() - start))
            try:
                self.save_rf_model(self.model, self.gh_filters, self.gh_features, self.gh_model_name)
                logger.info("Model saved to file: {}".format(self.gh_model_name + '.pkl'))
            except IOError:
                raise IOError("Could not save model to disk")


        if reco_type == 'en':
            self.model = RandomForestRegressor(n_estimators=self.en_rf_trees, max_features=self.en_rf_max_features,
                                               n_jobs=self.en_rf_jobs, oob_score=self.en_rf_oob_score,
                                               random_state=42)

            if self.used_preprocessed_data:
                with np.load("gammas_for_enreco_training.npz") as data:
                    self.features = data['f']
                    self.target = data['t']

            logger.info('Start training RF regressor for energy')
            start = time.process_time()
            self.model.fit(self.features, self.target)
            logger.info('...done. Training RF regressor for energy reconstruction took {:.2f} s'.\
                        format(time.process_time() - start))

            try:
                self.save_rf_model(self.model, self.en_filters, self.en_features, self.en_model_name)
                logger.info("Model saved to file: {}".format(self.en_model_name + '.pkl'))
            except IOError:
                raise IOError("Could not save model to disk")

        elif reco_type == 'dir':
            self.model = RandomForestRegressor(n_estimators=self.dir_rf_trees, max_features=self.dir_rf_max_features,
                                               n_jobs=self.dir_rf_jobs, oob_score=self.dir_rf_oob_score,
                                               random_state=42)

            if self.used_preprocessed_data:
                with np.load("gammas_for_dirreco_training.npz") as data:
                    self.features = data['f']
                    self.target = data['t']

            logger.info('Start training RF regressor for direction')
            start = time.process_time()
            self.model.fit(self.features, self.target)
            logger.info('...done. Training RF regressor for direction reconstruction took {:.2f} s'.\
                        format(time.process_time() - start))

            try:
                self.save_rf_model(self.model, self.dir_filters, self.dir_features, self.dir_model_name)
                logger.info("Model saved to file: {}".format(self.dir_model_name + '.pkl'))
            except IOError:
                raise IOError("Could not save model to disk")

    def print_feature_importances(self, reco_type='gh'):
        if reco_type == 'gh':
            logger.info("gh separation features importances:")
            for i in range(self.gh_num_features):
                logger.info("{} importance -> {:.3f}".format(self.gh_features[i],
                                                             self.model.feature_importances_[i]))
            logger.info("Out of bag (oob) score: {}".format(self.model.oob_score_))
        elif reco_type == 'en':
            logger.info("energy reconstruction features importances:")
            for i in range(self.en_num_features):
                logger.info("{} importance -> {:.3f}".format(self.en_features[i],
                                                             self.model.feature_importances_[i]))
            logger.info("Out of bag (oob) score: {}".format(self.model.oob_score_))
        elif reco_type == 'dir':
            logger.info("direction reconstruction features importances:")
            for i in range(self.dir_num_features):
                logger.info("{} importance -> {:.3f}".format(self.dir_features[i],
                                                             self.model.feature_importances_[i]))
            logger.info("Out of bag (oob) score: {}".format(self.model.oob_score_))


class EventRecoPredictor(object):
    def __init__(self, params):
        self.input_file = params['input_file']
        self.evtable = params['evtable']
        self.outfile = params['outfile']
        self.is_mc = params['is_mc']

        self.do_gh = params['do_gh_sep']
        self.do_en = params['do_en_reco']
        self.do_dir = params['do_dir_reco']

        self.gh_model_file = params['gh_model_file']
        self.en_model_file = params['en_model_file']
        self.dir_model_file = params['dir_model_file']

        self.head_tail = params['head_tail']

        self.force_cuts = params['force_cuts']

        # self.use_caldb = params['use_caldb']

        # empty init
        self.model = None
        self.filters = str()
        self.columns = list()
        self.gammaness = None
        self.en_reco = None
        self.disp_reco = None
        self.meanx = None
        self.meany = None
        self.delta = None
        self.ht_discriminator = None
        self.x_reco = None
        self.y_reco = None

    def load_rf_model(self, filename):
        try:
            model_info = joblib.load(filename)
            self.model = model_info['model']
            self.filters = model_info['filters']
            self.columns = model_info['feature_names']
        except IOError:
            raise IOError('Could not read model info from file, please check')

    def init_reco_params(self, nrows):
        self.gammaness = np.full(nrows, None)
        self.en_reco = np.full(nrows, None)
        self.disp_reco = np.full(nrows, None)
        self.x_reco = np.full(nrows, None)
        self.y_reco = np.full(nrows, None)

    @staticmethod
    def read_nrows(infile, evtable):
        with fitsio.FITS(infile) as hdu:
            data = hdu[evtable].read()
            return data.size

    def features_extractor(self, infile, evtable, filters, npar, reco_type='gh', *args, **kwargs):
        """Generates event list with npar number of features defined in
        *args. To see what args can be, refer to astriluts_scikit.par
        Since it yields objects, it has to be called with np.stack (and squeeze)
        """
        # this imports allowed math functions to be used in astriml.par
        from numpy import log10, cos, sin, abs

        with fitsio.FITS(infile) as hdu:

            mask = hdu[evtable].where(filters)

            # NEVER EVER CHANGE events VARIABLE NAME!!!
            # the events variable defined here is used in eval function
            # see astriluts.par
            events = hdu[evtable][mask]
            num_events = len(events)

            evts_feat = np.zeros(shape=(num_events, npar))

            for i in range(npar):
                evts_feat[:, i] = eval(args[i])

            self.features = evts_feat

            if reco_type == 'dir':
                self.meanx = events["MEANX"]
                self.meany = events["MEANY"]
                self.delta = events["DELTA"]
                self.ht_discriminator = events[self.head_tail]

    def predict(self, reco_type='gh'):

        if reco_type == 'gh':
            prob = self.model.predict_proba(self.features)
            self.gammaness = prob[:, 1]
        elif reco_type == 'en':
            self.en_reco = self.model.predict(self.features)
        else:
            self.disp_reco = self.model.predict(self.features)

    def srcpos_xy(self):

        self.x_reco = self.meanx - self.disp_reco * np.cos(self.delta) * np.copysign(1, self.ht_discriminator)
        self.y_reco = self.meany - self.disp_reco * np.cos(self.delta) * np.copysign(1, self.ht_discriminator)

    def equalize_array_size(self):

        largest_size = np.max([self.gammaness.size, self.en_reco.size, self.disp_reco.size])

        if self.gammaness.size < largest_size:
            fill = np.full(largest_size - self.gammaness.size, None)
            self.gammaness = np.concatenate((self.gammaness, fill))
        if self.en_reco.size < largest_size:
            fill = np.full(largest_size - self.en_reco.size, None)
            self.en_reco = np.concatenate((self.en_reco, fill))
        if self.disp_reco.size < largest_size:
            fill = np.full(largest_size - self.disp_reco.size, None)
            self.disp_reco = np.concatenate((self.disp_reco, fill))
            self.x_reco = np.concatenate((self.x_reco, fill))
            self.y_reco = np.concatenate((self.y_reco, fill))

    def write_lv1cfits(self):
        """This writes down the lv1c file in fits format
        - extend EVENTS extensions from lv1b to lv1c;
        - check if the input is MC or REAL;
        - copy (if present) SIM_EVENTS and TELCONF extensions to lv1c
        :return: True if success
        """
        new_cols = list()
        new_cols.append(fits.Column(name='GAMMANESS', format='E', unit="", array=self.gammaness))
        new_cols.append(fits.Column(name='EST_EN', format='E', unit="[TeV]", array=self.en_reco))
        new_cols.append(fits.Column(name='DISP', format='E', unit="[deg]", array=self.disp_reco))
        new_cols.append(fits.Column(name='XDISP', format='E', unit="[deg]", array=self.x_reco))
        new_cols.append(fits.Column(name='YDISP', format='E', unit="[deg]", array=self.y_reco))

        new_cols_fit = fits.ColDefs(new_cols)

        with fits.open(self.input_file) as lv1b:
            events = lv1b['EVENTS'].data.copy()

            evtable = fits.BinTableHDU.from_columns(events.columns + new_cols_fit)
            evtable.name = self.evtable

            # lv1c_dtypes = [('GAMMANESS', 'f4'), ('EST_EN', 'f4'), ('DISP', 'f4'), ('XDISP', 'f4'), ('YDISP', 'f4')]
            # new_dt = np.dtype(events.dtype.descr + lv1c_dtypes)
            # new_evts = np.full_like(events, events, dtype=new_dt)
            #
            # # now fill with new values
            # new_evts['GAMMANESS'] = self.gammaness
            # new_evts['EST_EN'] = self.en_reco
            # new_evts['DISP'] = self.disp_reco
            # new_evts['XDISP'] = self.x_reco
            # new_evts['YDISP'] = self.y_reco

            if self.is_mc:
                sim_events = lv1b['SIM_EVTS'].data.copy()
                simtable = fits.BinTableHDU.from_columns(sim_events.columns)
                simtable.name = "SIM_EVTS"

                telconf = lv1b['TELCONF'].data.copy()
                teltable = fits.BinTableHDU.from_columns(telconf.columns)
                teltable.name = "TELCONF"

                hdulist = fits.HDUList([fits.PrimaryHDU(), evtable, simtable, teltable])
            else:
                hdulist = fits.HDUList([fits.PrimaryHDU(), evtable])


            hdulist.writeto(self.outfile, overwrite=True)


                # copy these to lv1c file
