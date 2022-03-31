from src.utils.summarize_graphs import ven_diagram, scatter_plot_predictions, correlation_graph, creating_training_time, average_correlations

target = 'ace'
targets = ['ace', 'spike', '']
primary_targets = ['target3']
models = ['lstm', 'decision_tree', 'sgdreg', 'xgboost']
descriptor = 'morgan_onehot_mac_circular'
labels = ['7k', '50k', '350k']
training_sizes = [7000, 50000, 350000]
labels_for_times = ['lstm_7k', 'lstm_50k', 'lstm_350k', 'decision_tree_7k',
                    'decision_tree_50k', 'decision_tree_350k',
                    'sgdreg_7k', 'sgdreg_50k', 'sgdreg_350k',
                    'xgboost_7k', 'xgboost_50k', 'xgboost_350k']
 #scatter_plot_predictions(targets, models, descriptor, 7000)
 #correlation_graph('ace', models, labels, training_sizes)
#average_correlations(models, training_sizes, primary_targets)
creating_training_time(training_sizes, labels_for_times, models,primary_targets)
ven_diagram(primary_targets,models)

# logger.info(f"Generating Results Has Started")
# # summarize_results(args.targets, args.regressors, args.training_sizes)
# logger.info(f"Generating Results Has Ended")
# targets = ['ace', 'spike']
# models = ['decision_tree', 'swift_dock']
# result = get_more_a_specific_result(targets, models, "7000", "morgan_onehot_mac_circular", "test_mse")
