import logging
import sys
from pyAstriPar import ParameterSystem
from astriml_classes import EventsRecoTrainer
from pprint import pprint
import time
# import pdb


logger = logging.getLogger('astriluts')

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

    event_reconstructor = EventsRecoTrainer(params)

    # logs all object attributes
    logger.info("Parameters read:")
    pprint(vars(event_reconstructor))

    # fetch data, wrangle them and train model

    # train model for GH SEPARATION
    if event_reconstructor.do_gh:
        if not event_reconstructor.used_preprocessed_data:
            logger.info("Setup data for training gh separation")
            start = time.process_time()

            event_reconstructor.setup_for_training(reco_type='gh')  # after this you have features and target available

            logger.info("... done. Data wrangling took: {:.2f}".format(time.process_time() - start))
        else:
            logger.info('Reading preprocessed data from file')

        event_reconstructor.train(reco_type='gh')
        event_reconstructor.print_feature_importances(reco_type='gh')

    # train model for ENERGY RECONSTRUCTION
    if event_reconstructor.do_en:
        if not event_reconstructor.used_preprocessed_data:
            logger.info("Setup data for training energy reconstructor")
            start = time.process_time()

            event_reconstructor.setup_for_training(reco_type='en')

            logger.info("... done. Data wrangling took: {:.2f}".format(time.process_time() - start))
        else:
            logger.info('Reading preprocessed data from file')

        event_reconstructor.train(reco_type='en')
        event_reconstructor.print_feature_importances(reco_type='en')

    # train model for DIRECTION RECONSTRUCTION
    if event_reconstructor.do_dir:
        if not event_reconstructor.used_preprocessed_data:
            logger.info("Setup data for training direction reconstructor")
            start = time.process_time()

            event_reconstructor.setup_for_training(reco_type='dir')

            logger.info("... done. Data wrangling took: {:.2f}".format(time.process_time() - start))
        else:
            logger.info('Reading preprocessed data from file')

        event_reconstructor.train(reco_type='dir')
        event_reconstructor.print_feature_importances(reco_type='dir')

if __name__ == '__main__':
    main()
