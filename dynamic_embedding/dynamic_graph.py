import datetime
import math
import traceback

from tqdm import tqdm

from dynamic_embedding.sampler import NodeSampler
from utils.write_log import write_log
from utils.utils import *

import igraph as ig
from igraph import Graph
from tqdm import tqdm
import numpy as np
import math

app_debug = write_log("pipeline/logging", "debug", "dynamic_graph")


class DynGraphIgraph:
    def __init__(self, prefixes=None, flatten=[], directed=False, smooth=None):
        self.graph = ig.Graph(directed=directed)
        self.node_classes = {}  # prefix -> class ID (for node_class logic) ex. {'idx': 5, 'cid': 0}
        self.node_is_numeric = {} # ex. {'idx' : false, 'cid': false}
        self.to_flatten = flatten if flatten != 'all' else []
        self.dyn_roots=set()
        self.samplers = []

        # init records number (idx)
        self.graph['num_ids'] = -1

        # set smoothing method
        self.graph['smooth'] = smooth

        # set edge weight
        self.graph['weighted'] = directed

        if prefixes:
            self._extract_prefix(prefixes)
            self._check_flatten()

    def _extract_prefix(self, prefixes):
        for prefix in prefixes:
            class_info, name = prefix.split('__')
            rwclass = int(class_info[0])
            num_flag = class_info[1]
            self.node_classes[name] = rwclass
            self.node_is_numeric[name] = True if num_flag == '#' else False

    def _check_flatten(self):
        if self.to_flatten and self.to_flatten != 'all':
            for prefix in self.to_flatten:
                if prefix not in self.node_classes:
                    raise ValueError(f'Unknown flatten type: {prefix}')

    def _get_node_type(self, node_name):
        for pre in self.node_classes:
            if node_name.startswith(pre + '__'):
                return pre
        raise ValueError(f'Node {node_name} does not match any known prefix.')

    def clean_attributes(self):
        '''get a version of igraph which is ready to be written in a file
        :return self.graph (iGraph)
        '''
        allowed_types = (str, int, float, bool)

        for v in self.graph.vs:
            for attr in list(v.attributes()):
                if not isinstance(v[attr], allowed_types):
                    del v[attr]

        for e in self.graph.es:
            for attr in list(e.attributes()):
                if not isinstance(e[attr], allowed_types):
                    del e[attr]
        return self.graph
    
    def load_graph(self, graph_file):
        self.graph = Graph.Read_GraphML(graph_file)
        self.graph['num_ids'] = int(float(self.graph['num_ids']))
        self._extend_sampler(self.graph.vcount())
        # rebuild vertex attributes
        for v in self.graph.vs:
            v['node_class'] = self._update_node_class(v["type"])
        for v in self.graph.vs:
            self._update_neighbors(v.index)
            v["test_pretraining"] = True
            v['test_neighbors_freq']={}

    def set_id_nums(self, id_num):
        self.graph['num_ids'] = id_num
    
    def accum_id_nums(self):
        self.graph['num_ids'] += 1
    
    def get_id_nums(self):
        return self.graph['num_ids']
    
    def get_smooth_method(self):
        return self.graph['smooth']
    
    def get_directed_info(self):
        return self.graph.is_directed()
  
    def _update_node(self, node_name, node_prefix):
        '''update vertex
        method1: regroup vertex values for all types, if value exists, return vertex index; if not, create a new vertex and return the newly created vertex index
        method2: regroup vertex only for attributes(col name), and create new vertex for each instance even its value exists already in the graph
        : param node_name(str): vertex value
        : param node_prefix(str): values in list ["rid", "tt", "tn", "cid"]

        : return node.index(int): the vertex index in the graph involved in the update process
        '''

        if self.graph.vcount() == 0:
            node = self._add_vertex(node_name, node_prefix)
        else:

        # ## method1: if the node.name exists, skip 
        #     try:
        #         node = self.graph.vs.find(name=node_name)
        #     except ValueError:
        #         ## if the node.name doesn't exist, create new vertex
        #         node = self._add_vertex(node_name, node_prefix)           
        # ## END method1

        ## method2: add new node anyway
            if node_prefix == "cid":
                try:
                    node = self.graph.vs.find(name=node_name)
                    node["frequency_in_graph"] = node["frequency_in_graph"] + 1
                    return node.index
                except ValueError:
                    ## if the cid node.name doesn't exist, create new vertex
                    node = self._add_vertex(node_name, node_prefix)
            else:
                node = self._add_vertex(node_name, node_prefix)
        ## END method2

        node["frequency_in_graph"] = node["frequency_in_graph"] + 1
        return node.index
                
    def _update_token(self, ins_value, prefix):
        if self.graph.vcount() == 0:
            node = self._add_vertex(ins_value, prefix)
        else:
        ## if the token value exists, skip 
            try:
                node = self.graph.vs.find(name=ins_value)
            except ValueError:
                ## if the node.name doesn't exist, create new vertex
                node = self._add_vertex(ins_value, prefix)
        node["frequency_in_graph"] = node["frequency_in_graph"] + 1
        return node.index           

    def _tokenization(self, value):
        '''tokenization by '_' for now 
        if you change the tokenization rules, ajust function "def clean_str(value: str)" in flie ytils.py
        '''
        return value.split('_')

    def _add_vertex(self, node_name, node_prefix):
        
        # add to graph
        node = self.graph.add_vertex(
            name=node_name,
            type=node_prefix,
            numeric=self.node_is_numeric.get(node_prefix, False),  # properties of value
            node_class=self._update_node_class(node_prefix),       # properties for sample
            appearing_frequency=0,
            frequency_in_graph = 1,
            test_pretraining=False,
            test_neighbors_freq={}
        )
    
        if node["node_class"]['isroot']:
            self.dyn_roots.add(node.index)
        return node

    def _update_node_class(self, prefix):
        # get some attributes for random walk
        node_class_bin = '{:03b}'.format(self.node_classes[prefix])
        node_class_dict = {
            'isfirst': bool(int(node_class_bin[0])),
            'isroot': bool(int(node_class_bin[1])),
            'isappear': bool(int(node_class_bin[2]))
        }
        return node_class_dict
    
    def _update_neighbors(self, index):        
        # store all neighbors' name in a list as an attribute of node with "node_name"
        v = self.graph.vs[index]
        neighbors = self.graph.neighbors(index, mode='OUT') # all edges are added in two direction, so we get the same result for mode "in"/"out"/"all"

        if not self.graph.is_directed():
            sampler = NodeSampler(neighbors=neighbors, weighted=False, threshold=1000)
                      
        else:
            weights = []
            for el in neighbors:
                eid = self.graph.get_eid(v, el)
                w = self.graph.es[eid]['weight'] if 'weight' in self.graph.es[eid].attributes() else 1.0
                weights.append(w)
            sampler = NodeSampler(neighbors=neighbors, weighted=True, threshold=50, weights=weights)

        if not v["node_class"]["isfirst"]:
            sampler.update_firstnode_list([self.graph.vs[idx]["node_class"]["isfirst"] for idx in neighbors])

        # add or overwrite sampler
        self.samplers[index] = sampler

    def get_sampler(self, index):
        if index < len(self.samplers):
            return self.samplers[index]
        return None
    
    def _add_edge(self, node1_index, node2_index):
        if not self.graph.is_directed():
            if not self.graph.are_connected(node1_index, node2_index):
                self.graph.add_edge(node1_index, node2_index)
        else:
            # directed and weighted, add edges bi-directionally with different weight
            self._add_edge_weight(from_index=node1_index, to_index=node2_index)
            self._add_edge_weight(from_index=node2_index, to_index=node1_index)

    def _add_edge_weight(self, from_index, to_index):

        # calculate vertex degree and edge weight if necessary
        degree = self.graph.degree(to_index, mode='OUT')  # option: 'OUT', 'IN'
        weight = self._get_weight(degree)
        # update edge weight
        if not self.graph.are_connected(from_index, to_index):
            self.graph.add_edge(from_index, to_index, weight=weight)
        else:
            # get edge id
            eid = self.graph.get_eid(from_index, to_index)
            # update weight for this id
            self.graph.es[eid]['weight'] = weight

    def _get_weight(self, degree):
        # smooth log
        if self.graph["smooth"] == "log":
            weight = 1 / (math.log(degree + 1) + 1)
        else:
            weight = degree
        return weight

    def _update_instance_vertex_edge(self, cid_index, rid_index, ins_value, prefix):
        '''for each instance, update vertex information and edges involved
        '''
        instances_index = set()

        instance_index = self._update_node(ins_value, prefix)
        instances_index.add(instance_index)
        # update vertex
        # if tokenization
        if prefix in self.to_flatten:
            valsplit = self._tokenization(ins_value)
            for val in valsplit:
                if val.strip() == "":
                    continue
                val_index = self._update_token(val, prefix)
                instances_index.add(val_index)

        for index in instances_index:
            # update edge
            self._add_edge(index, cid_index)
            self._add_edge(index, rid_index)
        return instances_index

    def build_relation(self, df):
        ''' update graph without deduplication of instances
        :param df: data in formet of dataframe
        '''
        affected_nodes = set()
        # Iterate over all rows in the df
        for _, df_row in tqdm(df.iterrows(), total=len(df), desc="# Building/Updating graph"):
            # get row id and update node
            rid_node = str(df_row['rid'])
            rid_index = self._update_node(rid_node, "idx")

            affected_nodes.add(rid_index)
            # Remove nans from the row
            row = df_row.dropna()
            # Create a node for the current row id.
            for cid_node in df.columns:

                if cid_node != "rid":
                    try:
                        # get attribute and update node
                        cid_index = self._update_node(cid_node, "cid")
                        affected_nodes.add(cid_index)

                        # get instance and update node
                        og_value = row[cid_node]
                        # Convert cell values to strings, None or list.
                        token_list, is_numeric = convert_token_value(og_value)

                        for el in token_list:
                            if is_numeric:
                                node_prefix = "tn"
                            else:
                                node_prefix = "tt"
                            instance_index = self._update_instance_vertex_edge(cid_index, rid_index, el, node_prefix)
                            affected_nodes.update(instance_index)
                    except KeyError:
                        continue
        
        # extend sampler list
        self._extend_sampler(self.graph.vcount())
        # update neighbors
        for index in affected_nodes:
            self._update_neighbors(index)
            ## build roots list for graph
            # if self.graph.vs[index]["node_class"]['isroot']:
            #     self.dyn_roots.add(index)
    
    def _extend_sampler(self, num):
        """Expand the list for sampler so that it supports up to index"""
        if num >= len(self.samplers):
            # Add None to populate the expansion list
            self.samplers.extend([None] * (num + 1 - len(self.samplers)))
        
    def get_graph(self):
        return self.graph

    def get_neighbors(self, node_name):
        if node_name not in self.node_idx_map:
            return []
        idx = self.graph.vs.find(name=node_name)
        neighbors = self.graph.neighbors(idx, mode='ALL')
        return [self.graph.vs[n]['name'] for n in neighbors]
    
    def show_summary(self):
        print(self.graph.summary())

    def get_node_attribute(self, node_name, attr):
        return self.graph.vs.find(name=node_name)[attr]

    def set_node_attribute(self, node_name, attr, value):
        self.graph.vs.find(name=node_name)[attr] = value

def dyn_graph_generation(configuration):
    """
    Generate the graph for the given dataframe following the specifications in configuration.
    :param df: dataframe to transform in graph.
    :param configuration: dictionary with all the run parameters
    :return: the generated graph
    """
    if 'flatten' in configuration and configuration['flatten']:
        if configuration['flatten'].lower() not in ['all', 'false']:
            flatten = configuration['flatten'].strip().split(',')
        elif configuration['flatten'].lower() == 'false':
            flatten = []
        else:
            flatten = 'all'
    else:
        flatten = []

    t_start = datetime.datetime.now()
    print(OUTPUT_FORMAT.format('Starting graph construction', t_start.strftime(TIME_FORMAT)))

    prefixes = configuration["node_types"]
    directed = configuration["directed"]
    smooth = configuration["smoothing_method"]
    g = DynGraphIgraph(prefixes=prefixes, flatten=flatten, directed=directed, smooth=smooth)
    t_end = datetime.datetime.now()
    dt = t_end - t_start
    print()
    print(OUTPUT_FORMAT.format('Graph construction complete', t_end.strftime(TIME_FORMAT)))
    print(OUTPUT_FORMAT.format('Time required to build graph:', f'{dt.total_seconds():.2f} seconds.'))
    return g