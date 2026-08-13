# Hierarchical Unsupervised Efficient Coding for Neural Alignment

This repository provides code to initialize a model using hierarchical unsupervised efficient coding, perform supervised fine-tuning, and evaluate the model on brain alignment using the Natural Scenes Dataset (NSD).

# The following environmental variables need to be set

NSD_DATA_PATH : e.g. ".../for_atlas"

NSD_STIMULI_PATH: e.g. '.../nsd_stimuli.hdf5'

BACKPROP_TRAINING_PATH : where all backrpop runs are saved, e.g.  '.../bottleneck_training_runs'

ENCODING_EVAL_PATH : where all encoding evaluation results will be saved, e.g.  '.../deep_bottleneck_data/encoding_eval'

ENCODING_CACHE_PATH : where cached files should be saved to speed up evaluation, e.g. '.../deep_bottleneck_data/encoding_cache'

# Usage

# Unsupervised Learning

Train the model using hierarchical unsupervised efficient coding:

```bash
python training_alternate_layers.py
```

# Supervised Fine-tuning

Launch supervised training using the provided shell script and configuration files:

```bash
bash launch_scat_training.sh
```

Configuration options:

training_params_pca.json — PCA-based initialization
training_params_rand.json — Random initialization
