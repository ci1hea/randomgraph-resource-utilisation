import os
import time
import logging
import numpy as np
import pandas as pd
import networkx as nx
# Own imports
from harpoon_maker.graphs import directed_gnm_random_graph
FORMAT = '[%(processName)s, %(process)d, %(threadName)s, %(asctime)s, %(levelname)s] %(message)s'


class GraphMaker(object):
    def __init__(self, data_dir, n_start=2, n_end=5, id=None):
        self.data_dir = data_dir
        self.n_start = n_start  # List of Node counts. Min value = 2
        self.n_end = n_end
        self.id = id
        logging.info('Initialised GraphMaker')
    
    def mc_make_graphs(self, log_lock, mc_id):
        logging.basicConfig(filename='logfile-{}.log'.format(self.id), level=logging.DEBUG, format=FORMAT)
        with log_lock:
            logging.info('Making graphs for MC id: {}'.format(mc_id))
        time_start = time.perf_counter()

        node_list = []
        manual = False
        if manual:
            alist = [1, 10]
            for i in range(len(alist) - 1):
                node_list += list(range(alist[i], alist[i + 1], alist[i]))
            node_list = node_list[1:]
        else:
            node_list = list(range(self.n_start, self.n_end + 1))

        mc_data = []  # Data repo
        for n in node_list:  # Looping through node counts
            
            max_edges = n*(n-1)
            for m in range(max_edges, max_edges+1):  # Looping through connectance
                UM_G = directed_gnm_random_graph(n, m)  # Init graph

                # Comment out for the validation of Theorem 1
                lambdas = np.random.uniform(low=0.01, high=0.99, size=UM_G.number_of_nodes())
                phis = np.random.uniform(low=0.01, high=0.99, size=UM_G.number_of_nodes())
                flows = np.random.uniform(low=0, high=1, size=UM_G.number_of_edges())
                
                # Uncomment for the validation of Theorem 1
                # l_same = np.random.uniform(low=0.01, high=0.99, size=1)
                # p_same = np.random.uniform(low=0.01, high=0.99, size=1)
                # f_same = np.random.uniform(low=0, high=1, size=1)
                # lambdas = [l_same for x in range(1, UM_G.number_of_nodes()+1)]
                # phis = [p_same for x in range(1, UM_G.number_of_nodes()+1)]
                # flows = [f_same for x in range(1, UM_G.number_of_edges()+1)]

                nx.set_node_attributes(UM_G, dict(zip(range(UM_G.number_of_nodes()), lambdas)), 'Lambda')  # Assigning random efficiency coefficient Lambda to each node
                nx.set_node_attributes(UM_G, dict(zip(range(UM_G.number_of_nodes()), phis)), 'Phi')  # Assigning random efficiency coefficient Phi to each node
                nx.set_edge_attributes(UM_G, dict(zip(UM_G.edges(), flows)), 'F')  # Assigning random flow value F to each edge

                true_p = float(len(UM_G.edges))/(float(n)*(float(n)-1))  # calculating true connectance for network assembled
                
                flow_import_total = 0
                flow_export_total = 0
                flow_destroyed_total = 0
                import_count = 0
                export_count = 0
                neutral_count = 0
                for i in range(n):  # Looping through nodes
                    Fin = 0   # Init node inflows
                    Eff = 0   # Init node efficiency
                    Fout = 0  # Init node outflows
                    Fd = 0    # Init node destroyed exergy
                    
                    Fin = [x for (a, b, x) in UM_G.in_edges(i, 'F')]  # Listing all node inflows
                    Eff = [((1-UM_G.node[source_node]['Lambda'])/(UM_G.node[source_node]['Lambda']*UM_G.node[source_node]['Phi']))
                           for source_node in [a for (a, b, x) in UM_G.in_edges(i, 'F')]]  # Listing all incoming flow nodes efficiencies
                    Fd = [a*b for a, b in zip(Fin, Eff)]  # Listing exergy destroyed per incoming flow
                    Fout = [x/(UM_G.node[i]['Lambda']*UM_G.node[i]['Phi']) for (a, b, x) in UM_G.out_edges(i, 'F')]  # Listing all node outflows
                    
                    balance = (np.sum(Fout)-np.sum(Fin))
                    if balance > 0:     # Checking whether node flow balance requires import or export
                        flow_import_total += (np.sum(Fout)-np.sum(Fin))  # Assigning import
                        import_count += 1
                    elif balance < 0:
                        flow_export_total += (np.sum(Fout)-np.sum(Fin))  # Assigning export
                        export_count += 1
                    else:
                        neutral_count += 1

                    flow_destroyed_total += np.sum(Fd)  # Summing total destroyed exergy
                
                mc_data.append([int(mc_id), n, m, round(true_p, 5), round(flow_destroyed_total, 5), round(flow_import_total, 5), round(flow_export_total, 5), import_count,
                                export_count, neutral_count, 'U(0;1)', round(np.mean(lambdas), 5), 'U(0;1)', round(np.mean(phis), 5), '1', round(np.mean(flows), 5)])  # Appending UMG iteration
        
        labels = ['mc_id', 'node_count', 'm', 'true_p', 'flow_destroyed_total', 'flow_import_total', 'flow_export_total',
                  'import_count', 'export_count', 'neutral_count', 'Lambda_dist', 'Lambda_mean', 'Phi_dist', 'Phi_mean', 'Flow_dist', 'Flow_mean']
        mc_data_df = pd.DataFrame.from_records(mc_data, columns=labels)
        output_filename = os.path.join(self.data_dir, 'gnm_di_graph_mc{}.csv'.format(mc_id))
        mc_data_df.to_csv(output_filename, index=False)  # Save out

        with log_lock:
            time_end = time.perf_counter()
            logging.info('Finished making graphs for MC id: {} in {} seconds'.format(mc_id, time_end - time_start))