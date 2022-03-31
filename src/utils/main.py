from src.utils.summarize_graphs import ven_diagram, \
    scatter_plot_predictions, \
    correlation_graph, \
    creating_training_time, \
    average_correlations, summarize_results, get_more_a_specific_result

target = 'ace'
targets = ['ace', 'spike', 'nsp', 'nsp_sam']
primary_targets = ['target3']
models = ['lstm', 'decision_tree', 'sgdreg', 'xgboost']
descriptor = 'morgan_onehot_mac_circular'
labels = ['7k', '10k', '20k']
training_sizes = [7000, 50000, 350000]
training_sizes_other = [7000, 10000, 20000]
labels_for_times = ['lstm_7k', 'lstm_50k', 'lstm_350k', 'decision_tree_7k',
                    'decision_tree_50k', 'decision_tree_350k',
                    'sgdreg_7k', 'sgdreg_50k', 'sgdreg_350k',
                    'xgboost_7k', 'xgboost_50k', 'xgboost_350k']
training_size = 7000
tok_k = 10000
list_of_labels = []
# creating_training_time(training_sizes, models, primary_targets)
# ven_diagram(primary_targets, models, training_size, descriptor, tok_k)
#scatter_plot_predictions(targets, models, descriptor, 7000)
#correlation_graph('spike', models, labels, training_sizes_other)
#average_correlations(models, training_sizes, primary_targets)
summarize_results(targets, models, training_sizes_other)
# result = get_more_a_specific_result(targets, models, "7000", "morgan_onehot_mac_circular", "test_mse")
