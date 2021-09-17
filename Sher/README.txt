analyze_predictions.py

"preprocess_inputdata.py" processes the post-processed output of PUMA to a format the neural network can use directly

"network_tuning.py"   tunes the network

"train_NN.py" does the training of the network (with the configuration from the tuning)

"analyze_predictions.py" analyzes the results of the training

"NN_climate.py" makes a network-climate run with the trained network

the "puma" directory contains namelists and postprocessing files for PUMA

"trained_network_weights" contains the weights of the trained network


The file 'puma_sample_30year_normalized.nc' is a 30-year sample of the PUMA run. It is outside the spinup-period and can therefore be used completely.


