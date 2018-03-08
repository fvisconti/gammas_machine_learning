import logging
import sys
from pyAstriPar import ParameterSystem
from astriml_classes import EventRecoPredictor
from pprint import pprint
import time
# from numpy import savetxt, c_, float32
# import pdb


logger = logging.getLogger('astrireco')
try:
    import colorlog

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s (%(process)s) %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
except ImportError:
    logging.basicConfig(format="%(asctime)s (%(process)s) %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger.setLevel(logging.INFO)

def main():
    # get parameters
    params = ParameterSystem(sys.argv)

    reco = EventRecoPredictor(params)

    # logs all object attributes
    logger.info("Parameters read:")
    pprint(vars(reco))

    nrows = reco.read_nrows(reco.input_file, reco.evtable)

    # init reco params to array of None with length nrows
    reco.init_reco_params(nrows)

    # train model for GH SEPARATION
    if reco.do_gh:
        logger.info("Predict gammaness")
        start = time.process_time()

        reco.load_rf_model(reco.gh_model_file)
        reco.features_extractor(reco.input_file, reco.evtable, reco.filters, len(reco.columns), 'gh', *reco.columns)
        reco.predict(reco_type='gh')  # after this you have features and target available

        logger.info("... done. Gammaness prediction took: {:.2f}".format(time.process_time() - start))

        logger.info("Other reconstruction parameters:")
        print("Filters: "), pprint(reco.filters)
        print("Model: "), pprint(reco.model)
        print("Columns :"), pprint(reco.columns)

    # train model for ENERGY RECONSTRUCTION
    if reco.do_en:
        logger.info("Predict energy")
        start = time.process_time()

        reco.load_rf_model(reco.en_model_file)
        reco.features_extractor(reco.input_file, reco.evtable, reco.filters, len(reco.columns), 'en', *reco.columns)
        reco.predict(reco_type='en')

        logger.info("... done. Energy prediction took: {:.2f}".format(time.process_time() - start))

        logger.info("Other reconstruction parameters:")
        print("Filters: "), pprint(reco.filters)
        print("Model: "), pprint(reco.model)
        print("Columns :"), pprint(reco.columns)

    # train model for DIRECTION RECONSTRUCTION

    if reco.do_dir:
        logger.info("Predict direction")
        start = time.process_time()

        reco.load_rf_model(reco.dir_model_file)
        reco.features_extractor(reco.input_file, reco.evtable, reco.filters, len(reco.columns), 'dir', *reco.columns)
        reco.predict(reco_type='dir')
        reco.srcpos_xy()

        logger.info("... done. Direction prediction took: {:.2f}".format(time.process_time() - start))

        logger.info("Other reconstruction parameters:")
        print("Filters: "), pprint(reco.filters)
        print("Model: "), pprint(reco.model)
        print("Columns :"), pprint(reco.columns)


    reco.write_lv1cfits()

    # logger.info("Saving reconstructed quantities to text file: {}".format(reco.outfile))
    # # equalize sizes
    # reco.equalize_array_size()
    # try:
    #     savetxt(reco.outfile, c_[float32(reco.gammaness), float32(reco.en_reco), float32(reco.disp_reco),
    #                              float32(reco.x_reco), float32(reco.y_reco)])
    # except IOError:
    #     raise IOError("Could not save file: {}".format(reco.outfile))

if __name__ == '__main__':
    main()
