# TITLE
## CONFIGURATION
There are 3 configuration files in dir `/config`. several blocs in each file. `mandatory`: config necessary to run the program; `optional config`: you can choose to personalize some config, if not, the program whill use values be default.


default values list:
| VARIABLE NAME    | VALUE BY DEFAULT |
| -------- | ------- |
| log_path  | "pipeline/logging"  |
| output_file_name  | "test" |
| node_types | ["7#__tn", "7$__tt", "3$__idx", "1$__cid"] |
| flatten |  "no" | 
| smoothing_method|  "no" | 
| directed| False | 
| write_walks|  True | 
| walk_length|  60 | 
| walks_number|  20 | 
| backtrack|  False | 
| rw_stat| True | 
| learning_method|  "skipgram" | 
| window_size|  3 | 
| n_dimensions|  300 | 
| training_algorithm|  "word2vec" | 
| sampling_factor |  0.001 |
| top_k| 10 | 
| simlist_n| 10 | 
| simlist_show| 5 | 
| strategy_suppl| "faiss" | 
| output_format| "db" | 
| source_num| 0 | 
| kafka_topicid| "entity_resolution_process" | 
| kafka_groupid| "er_group" | 
| bootstrap_servers| "localhost" | 
| port| "9092" | 
| window_strategy| "count" | 
| window_count| 136 | 
| update_frequency| 0| 
| n_first|  3 | 
| approximate|  16 | 
| sim_vis|  True | 
| source_num| 0| 
