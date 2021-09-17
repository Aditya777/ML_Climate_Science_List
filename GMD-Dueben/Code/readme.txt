This repository contains the code used for the paper "Challenges and design choices for global weather and climate models based on machine learning" by Peter Dueben and Peter Bauer.

Toy model for global dynamics:

1. ./Toy_model/ERA_retrieve_from_Mars contains the scripts used to retrieve the ERA data from MARS at ECMWF.

2. ./Toy_model/NN_training contains the python programmes used for training of the network. 
bulk_1field.py : Network for update of the bulk of the gridpoint for local networks for one field using different stencils. 
bulk_2fields.py : Network for update of the bulk of the gridpoint for local networks for two fields using different stencils. 
pole_1field.py : Network for update of poles for local networks for one field. 
pole_2fields.py : Network for update of poles for local networks for two fields. 
global.py : Network for update of the entire domain for global networks for one field.
global_2fields.py : Network for update of the entire domain for global networks for two fields.

3. ./Toy_model/NN_forecasts contains the python programmes used for forecasts with the network for local and global networks and for one and two input/output fields.


Three level Lorenz'95 model:

1. The reference solution and training data was calculated using the model published here with small changes:
Hatfield, S., 2017: samhatfield/lorenz96-ensrf: Publication (version v1.0.0). Zenodo, accessed 8 December 2017, https://doi.org/10.5281/zenodo.571203

2. ./Lorenz95 contains the two python programmes used for training and to perform the forecasts for the local and the global configuration.
