import json
import queue
from threading import Timer
import threading
import pandas as pd
import time
from kafka import KafkaConsumer
from collections import deque
import traceback

from prometheus_client import start_http_server
from tqdm import tqdm
from dynamic_embedding.dynamic_sampling import dynrandom_walks_generation
from dynamic_embedding.dynamic_entity_resolution import dynentity_resolution, FaissIndex
from dataprocessing.similaritylist import SimilarityList
from dataprocessing.metrics import Metrics
from dataprocessing.random_walk_analysis import random_walk_analysis
from utils.write_log import write_log

task_queue = queue.Queue()
    
class ConsumerService:
    def __init__(self, configuration, graph, model, output_file_name, prefixes, metrics):
        self.config = configuration
        self.graph = graph
        self.model = model
        self.strategy_suppl = configuration["strategy_suppl"]
        self.strategy_model = None
        self.output_file_name = output_file_name
        self.prefixes = prefixes
        self.sim_list = SimilarityList(
                            configuration["most_similar_inlist_n"],
                            configuration["output_format"]
                        )
        self.window_data = deque()
        self.last_update_time = time.time()
        self.data_buffer = []
        self.flag_running = False
        self.write_timer = None
        self.time_interval = 2 # 300 seconds = 5 minutes
        self.app_logger = None
        self.kafka_logger = None
        self.debug_logger = None
        self.metrics = metrics
        self._setup_logging()
        self.timeout = 60 # 60 (s)
        self.timer = None
        self.t_start_time = None
        self.t_end_time = None

    def _setup_logging(self):
        self.app_logger = write_log(self.config["log_path"], "app", "app")
        self.kafka_logger = write_log(self.config["log_path"], "kafka", "kafka")
        self.debug_logger = write_log(self.config["log_path"], "debug", "debug")

        self.sim_list.set_logger(self.app_logger)

    def trigger_file_write(self):
        self.sim_list.check_output_path(self.output_file_name)
        if self.flag_running:
            self.write_timer = Timer(self.time_interval, self.trigger_file_write)  
            self.write_timer.daemon = True  # Make it a daemon thread
            self.write_timer.start()
        self.sim_list.update_file()

    def _remove_expired_data(self, current_time):
        """Remove data older than the window size"""
        while self.window_data and (current_time - self.window_data[0]["timestamp"] > self.config["window_time"]):
            self.window_data.popleft()

    def _prepare_data(self, window_data):
        # construct data structucre
        data_list = list(window_data)
        df = pd.DataFrame.from_records(data_list)
        return df

    def build_matching_list(self, df):
        '''build sim list and output to db file'''
        # get similar words 
        for target in tqdm(df.loc[:,"rid"], desc= "# build similarity list. "):
            try:
                if self.strategy_suppl == "basic":
                    similar = dynentity_resolution(self.model, target, self.config["most_similar_k"])
                elif self.strategy_suppl == "faiss":
                    similar = self.strategy_model.get_similar_words([self.model.wv[target]], target, self.config["most_similar_k"])

                if int(self.config['source_num']) > 0 and similar != []:
                    similar = self._filter_list(similar)
                
                if similar != [] and similar is not None:
                    self.sim_list.add_similarity(target, similar)

                    if self.sim_list.output_format == "db":
                        # print(target)
                        self.sim_list.insert_data(target)
                    # print(self.sim_list.get_similarity_words_with_score(target, self.config["show_m_most_similar"]))
                    for word, score in similar:
                        self.sim_list.add_similarity(word, [(target, score)])
                        if self.sim_list.output_format == "db":
                            self.sim_list.insert_data(word)
                else:
                    pass
            except Exception as e:
                self.app_logger.error(f"Error similarity building: {str(e)}")
                self.debug_logger.error(traceback.print_exc())
                print(f"Error similarity building: {str(e)}")
        print()
                

    def process_window_data(self):
        """Process data of the current window """
        print("processing window data...")
        self.timer.cancel()
        
        # data preparation
        df = self._prepare_data(self.window_data)
        # add new node to graph
        self.graph.build_relation(df)
        print(f"# roots numbers: {len(self.graph.dyn_roots)}")
        # start random walk for new data
        walks = dynrandom_walks_generation(self.config, self.graph)

        if walks == []:
            raise ValueError(f"Random walk anomaly")

        try:
            print("# Retraining embeddings model by window data...")
            if len(self.model.wv.key_to_index) == 0:
                # initiate model
                self.model.build_vocab(walks)
                self.model.train(walks, total_examples=self.model.corpus_count, epochs=10)
                if self.strategy_suppl == "faiss":
                    self.strategy_model = FaissIndex(self.model)
            else:
                # Update model
                self.model.build_vocab(walks, update=True)
                self.model.train(walks, total_examples=len(walks), epochs=5) # An epoch is one complete pass through the entire training data.
                if self.strategy_suppl == "faiss":
                    if self.strategy_model == None:
                        self.strategy_model = FaissIndex(self.model)
                    else:
                        self.strategy_model.rebuild_index(self.model, self.graph.get_id_nums())
                        # self.strategy_model.update_index(self.model)
        except Exception as e:
            print("[ERROR]: ", e)
            self.app_logger.error(f"[ERROR]: {str(e)}")

        self.build_matching_list(df)

    def _reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(self.timeout, self._timerout_function)
        self.timer.start()

    def _timerout_function(self):
        print("last windows of this period")
        self.debug_logger.info("### last windows of this period... ")

        if len(self.window_data) != 0:
            self.sim_list.check_output_path(self.output_file_name)
            self.process_window_data()
            self.window_data.clear()
            self.t_end_time = time.time()
            print(f'[Finished] the test finished, num: {self.output_file_name}, execution time: {self.t_end_time - self.t_start_time}')
            self.app_logger.info(f'[Finished] the test finished, num: {self.output_file_name}, execution time (s): {round(self.t_end_time - self.t_start_time - 60)}')

            print(f"Saving model in binary format... Embedding file: pipeline/embeddings/{self.output_file_name}.emb")
            self.model.save(f"pipeline/embeddings/{self.output_file_name}.emb") # Saves in binary format by default
            # self.model.wv.save_word2vec_format(f"pipeline/embeddings/{self.output_file_name}.emb", binary=False) # Saves in binary format by default
            print("Model saved!")

            if self.config["rw_stat"] == True:
                print(f"drawing... Distribution of random walk visit values: {self.output_file_name}")
                random_walk_analysis(self.graph, self.output_file_name)
                print("Graph Ok!")

            print(f"Saving graph with attributes... Graph file: /home/zhongwei/Data_ingestion/embIng/pipeline/graph")
            g = self.graph.clean_attributes()
            g.write_graphml(f"pipeline/graph/graph-{self.output_file_name}.graphml")
            print("Graph saved!")
            print("Process over!!!!!!!!!!!!!!!!")

            

    def _filter_list(self, similarity_list):
        result = []
        if similarity_list is not None and similarity_list != []:
            for t in similarity_list:
                if int(float(t[0].split('__')[1])) <= int(self.config['source_num']):
                    result.append(t)
                    # print(t)
        return result



    def run(self):
        try:
            # prepare kafka consumer
            consumer = KafkaConsumer(
                self.config['kafka_topicid'],
                bootstrap_servers=f'{self.config["bootstrap_servers"]}:{self.config["port"]}',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=self.config["kafka_groupid"],
                enable_auto_commit=True
            )
        except Exception as e:
            self.app_logger.error(f"Fatal error in consumer service: {str(e)}")
            print(f"Fatal error in consumer service: {str(e)}")
            return

        self.app_logger.info("Start Kafka consumer...")

        while True:
            msg_pack = consumer.poll(timeout_ms=100)  # Non-blocking batch pull

            for tp, messages in msg_pack.items():
                for msg in messages:
                    if self.t_start_time is None:
                        self.app_logger.info("[STARTED] Receiving records...")
                        print("[STARTED] Receiving records...")
                        self.t_start_time = time.time()

                    try:
                        ## Update metrics
                        # Update lag
                        try:
                            latest_offset = consumer.end_offsets([tp])[tp]
                            lag = latest_offset - msg.offset
                            self.metrics.update_lag_metrics(lag)
                        except Exception as lag_err:
                            self.app_logger.warning(f"[Lag fetch error]: {lag_err}")
                        # Update record messages consumed
                        self.metrics.update_message_consumed()

                        ## prepare data structure
                        self.graph.accum_id_nums()
                        id_num = self.graph.get_id_nums()

                        metadata = msg.value
                        metadata["rid"] = f"idx__{id_num}"
                        current_time = time.time()

                        self.data_buffer.append(metadata)

                        ## Handle (time or count based) windowing
                        if self.config["window_strategy"] == "time":
                            self._handle_time_window(metadata, current_time)
                        elif self.config["window_strategy"] == "count":
                            self._handle_count_window(metadata)
                        

                    except Exception as e:
                        self.app_logger.error(f"Error processing message: {str(e)}")
                        traceback.print_exc()
                    
                    self._reset_timer()

            # prevent tight loop
            time.sleep(0.01)  

    def _handle_time_window(self, metadata, current_time):
        metadata["timestamp"] = current_time
        self.window_data.append(metadata)

        if current_time - self.last_update_time >= self.config["update_frequency"]:
            self._process_window()
            self.last_update_time = current_time
        
            # clear window
            if self.config["update_frequency"] != 0:
                self._remove_expired_data(current_time)
            else: 
                self.window_data.clear()

    def _handle_count_window(self, metadata):
        self.window_data.append(metadata)

        if len(self.window_data) >= self.config["window_count"]:

            self._process_window()

            # clear window
            if self.config["update_frequency"] != 0:
                for _ in range(self.config["update_frequency"]):
                    if self.window_data:
                        self.window_data.popleft()
            else:
                self.window_data.clear()

    def _process_window(self):
        '''
        update metric for window data processing time 
        '''        
        start = time.time()
        self.process_window_data()
        end = time.time()
        self.metrics.update_window_data_processing_time((end - start) / len(self.window_data))
            

def _start_prometheus_port():
    print("start thread prometheus..")
    start_http_server(8000)
                                
def start_kafka_consumer(configuration, graph, model, output_file_name, prefixes):
    metrics = Metrics()
    t = threading.Thread(target=_start_prometheus_port, daemon=True)
    t.start()

    consumer = ConsumerService(configuration, graph, model, output_file_name, prefixes, metrics)
    # check output path
    consumer.sim_list.check_output_path(output_file_name)
    if consumer.sim_list.output_format != "db":
            consumer.flag_running = True
            consumer.trigger_file_write()
    consumer.run()

if __name__ == '__main__':
    # consumer_service(configuration, graph, wv)
    pass
