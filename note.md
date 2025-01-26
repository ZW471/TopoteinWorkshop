# Adding secondary structure features
 - add a new feature type .yaml file in ./config/features
 - add the calculation in the ./features/node_features.py (compute_scalar_node_features)
 - register its dimension in the ./models/utils.py (get_input_dim)
 - remember to use your new feature configuration in the training arguments

# Adding my own model