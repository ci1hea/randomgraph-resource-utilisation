
import logging
import os
import time
import argparse
import glob
import numpy as np
import multiprocessing as mp
import pandas as pd
from functools import partial
from datetime import datetime

from harpoon_maker.graph_maker import GraphMaker

if __name__ == '__main__':
    run_time_start = datetime.utcnow().strftime('%m-%d-%H%M%S-%f')[:-1]

    FORMAT = '[%(processName)s, %(process)d, %(threadName)s, %(asctime)s, %(levelname)s] %(message)s'
    logging.basicConfig(filename='logfile-{}.log'.format(run_time_start), level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser(description='This is a program that will output csv data files about drawn Monte Carlo network samples.')
    parser.add_argument('-p', '--processes', default=2, type=int, help='Set number of parallel process to run')
    parser.add_argument('-s', '--samples', default=4, type=int, help='Set total number of Monte Carlo samples to run')
    parser.add_argument('-o', '--output', default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), help='Set the path to store the output folder')
    parser.add_argument('-l', '--lower_node', default=2, type=int, help='Set the inclusive lower node limit that the MC sampling will start from')
    parser.add_argument('-u', '--upper_node', default=5, type=int, help='Set the inclusive upper node limit that the MC sampling will run up to')
    args = parser.parse_args()

    dataset_name = "default"

    pool_size = args.processes
    mc_count = args.samples
    dir_path = args.output
    node_lower = args.lower_node
    node_upper = args.upper_node

    output_path = os.path.join(dir_path,'output-{}'.format(run_time_start))  # onced zipped, this may need to be changed to parent dir
    print('Output Path:', output_path)

    time_start = time.perf_counter()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    graph_maker = GraphMaker(data_dir=output_path, n_start=node_lower, n_end=node_upper, id=run_time_start)

    pool = mp.Pool(processes=pool_size, maxtasksperchild=1)
    inputs = list(range(mc_count))
    mng = mp.Manager()
    l = mng.RLock()
    partial_func = partial(graph_maker.mc_make_graphs, l)
    outputs = pool.map(partial_func, inputs)
    pool.close()
    pool.join()
    
    time_end = time.perf_counter()
    logging.info('Finished all MCs in {} seconds'.format(time_end - time_start))

    time_start = time.perf_counter()
    files = glob.glob(os.path.join(glob.escape(output_path), '*.csv'))
    logging.info('path: {}'.format(os.path.join(output_path, '*.csv')))
    logging.info('Number of files to merge: {}'.format(len(files)))
    hdf_path = os.path.join(os.path.dirname(output_path), 'mergedoutput-{}-{}.h5'.format(dataset_name, run_time_start))
    with pd.HDFStore(hdf_path, mode='w', complevel=5, complib='bzip2') as store:
        for filename in files:
            store.append('table_name', pd.read_csv(filename, sep=',', dtype={'true_p': np.float32, 'flow_destroyed_total': np.float32, 'flow_import_total': np.float32, 'flow_export_total': np.float32}, index_col=False), index=False)

    time_end = time.perf_counter()
    logging.info('Merged and compressed in {} seconds'.format(time_end - time_start))