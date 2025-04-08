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
                    raise ValueError(f"The first node of the sentence could not be found. Please check your prefix property settings.")
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

def dynrandom_walks_generation(configuration, graph):
    """
    Traverse the graph using different random walks strategies.
    :param configuration: run parameters to be used during the generation
    :param graph: graph generated starting from the input dataframe
    :return: the collection of random walks
    """
    sentences = []
    sentence_length = int(configuration["sentence_length"])
    backtrack = configuration["backtrack"]
    random_walks_per_node = configuration["random_walks_per_node"]

    roots_index = graph.dyn_roots
    

    # ########### Random walks ############
    sentence_counter = 0
    if random_walks_per_node > 0:
        pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index)*configuration['random_walks_per_node'])
        for root in roots_index:
            # if cell in intersection:
            ######## random walk for each node
            r = []
            for _r in range(random_walks_per_node):
                try:

                    w = RandomWalk(
                        graph,
                        root,
                        sentence_length,
                        backtrack
                    )
                except Exception as e:
                    print("node: ", _r)
                    print(e)
                    print(traceback.print_exc())
                    break
                
                if w.get_walk() != []:
                    r.append(w.get_walk())
                else:
                    raise ValueError(f"random walk anormal")

            if configuration["write_walks"]:
                if len(r) > 0:
                    ws = [" ".join(_) for _ in r]
                    s = "\n".join(ws) + "\n"
                    walk_info.info(s)
                else:
                    pass
            sentences += r
            sentence_counter += random_walks_per_node
           
            pbar.update(random_walks_per_node)
        pbar.close()

    graph.dyn_roots.clear()

    return sentences