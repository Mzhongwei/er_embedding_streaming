import random
from tqdm import tqdm
import traceback

from utils.utils import *
from utils.write_log import write_log

app_debug = write_log("pipeline/logging", "debug", "random_walk")
walk_info = write_log("pipeline/logging", "walks", "random_walk_record")

class RandomWalk:
    def __init__(
        self,
        graph,
        starting_node_index,
        sentence_len,
        backtrack
    ):
        i_graph = graph.get_graph()
        self.walk = []
        
        # find node and its neighbors
        starting_node = i_graph.vs[starting_node_index]
        starting_node_name = starting_node['name']
        # app_debug.info(f"root name: {starting_node_name}")
        # first step 
        if starting_node['node_class']['isfirst']:
            self.walk = [starting_node_name]
        else:
            try:
                sampler = graph.get_sampler(starting_node_index)
                first_node_indice = sampler.sample_firstnode()
                # first_node_indice = starting_node['sampler'].sample_firstnode()
                if first_node_indice is not None:
                    first_node_name = i_graph.vs[first_node_indice]['name']
                    self.walk = [first_node_name, starting_node_name]
                else:
                    raise ValueError(f"The first node of the sentence could not be found. Please check your node_types settings.")
            except Exception:
                print(f"The first node of the sentence could not be found. Please check node {starting_node_name}.")
                app_debug.error(f"The first node of the sentence could not be found. Please check node {starting_node_name}, index {starting_node_index}.")
                   
        if self.walk != []:
            current_node_indice = starting_node_index
            current_node_name = starting_node_name
            current_node = starting_node
            sentence_step = len(self.walk)
        else:
            return

        # the next steps
        while sentence_step < sentence_len:
            previous_node = current_node
            previous_node_index = current_node_indice
            sampler = graph.get_sampler(previous_node_index)
            current_node_indice = sampler.sample()
            # current_node_indice = current_node['sampler'].sample()
            if current_node_indice is None:
                raise ValueError(f'No neighbors')
            current_node = i_graph.vs[current_node_indice]
            current_node_name = current_node['name'] 

            if not backtrack and current_node_name == self.walk[-1]:
                continue
            if not current_node["node_class"]["isappear"]:
                continue
          
            self.walk.append(current_node_name)
            previous_node['appearing_frequency'] = previous_node['appearing_frequency'] + 1
            if current_node_name in previous_node["test_neighbors_freq"]:
                previous_node["test_neighbors_freq"][current_node_name] = previous_node["test_neighbors_freq"][current_node_name] + 1
            else:
                previous_node["test_neighbors_freq"][current_node_name] = 1
            sentence_step += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]

class RandomWalk_MetaPath:
    def __init__(
        self,
        graph,
        starting_node_index,
        sentence_len,
        meta_path
    ):
        i_graph = graph.get_graph()
        self.walk = []
        # find node and its neighbors
        current_node = i_graph.vs[starting_node_index]
        current_node_indice = starting_node_index
        self.walk.append(current_node['name'])
        # init for the second node
        meta_index = 1 
        
        # the next steps
        sentence_step = len(self.walk)
        while sentence_step < sentence_len:
            try:
                next_type = meta_path[meta_index % len(meta_path)]
                sampler = graph.get_sampler(current_node_indice)
                next_node_indice = random.choice(sampler[next_type])
                next_node = i_graph.vs[next_node_indice]
                self.walk.append(next_node['name'])
                current_node = next_node
                current_node_indice = next_node_indice
            except Exception:
                    print(f"NO neighbors found for node: {current_node_indice}, name: {current_node['name']}. Looking for node type: {next_type}")
                    app_debug.error(f"NO neighbors found for node: {current_node}, name: {current_node['name']}. Looking for node type: {next_type}")
                    break
            sentence_step += 1
            meta_index += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]

def start_walk(roots_index, graph, walks_number, walk_length, write_walks, walk_rules):
    sentences = []
    sentence_counter = 0
    pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index)*walks_number)
    for root in roots_index:
        # if cell in intersection:
        ######## random walk for each node
        walks = []
        for _r in range(walks_number):
            try: 
                if isinstance(walk_rules, bool):
                    w = RandomWalk(
                        graph,
                        root,
                        walk_length,
                        walk_rules
                    )
                else: 
                    w = RandomWalk_MetaPath(
                        graph,
                        root,
                        walk_length,
                        walk_rules
                    )
            except Exception as e:
                print("node: ", _r)
                print(e)
                print(traceback.print_exc())
                break
            
            if w.get_walk() != []:
                walks.append(w.get_walk())
            else:
                raise ValueError(f"random walk anormal")

        if write_walks:
            if len(walks) > 0:
                ws = [" ".join(_) for _ in walks]
                s = "\n".join(ws) + "\n"
                walk_info.info(s)
            else:
                pass
        sentences += walks
        sentence_counter += walks_number

        pbar.update(walks_number)
    pbar.close()
    return sentences

def dynrandom_walks_generation(configuration, graph):
    """
    Traverse the graph using different random walks strategies.
    :param configuration: run parameters to be used during the generation
    :param graph: graph generated starting from the input dataframe
    :return: the collection of random walks
    """
    
    walk_length = int(configuration['walks']['walk_length'])
    backtrack = configuration['walks']['backtrack']
    walks_number = configuration['walks']['walks_number']
    meta_path = configuration['graph']['meta_path']
    write_walks = configuration['walks']['write_walks']

    if walks_number > 0:
        ############ Random walks ############
        if not meta_path:
            roots_index = graph.dyn_roots
            sentences = start_walk(roots_index, graph, walks_number, walk_length, write_walks, backtrack)
            graph.dyn_roots.clear()
        else:
            if isinstance(meta_path, list):
                sentences = []
                for path in meta_path:
                    roots_index = graph.dyn_roots[path[0]]
                    sentences += start_walk(roots_index, graph, walks_number, walk_length, write_walks, path)
                    graph.dyn_roots[path[0]].clear()
            else:
                roots_index = graph.dyn_roots[meta_path[0]]
                sentences = start_walk(roots_index, graph, walks_number, walk_length, write_walks, meta_path)
                graph.dyn_roots[meta_path[0]].clear()
    return sentences