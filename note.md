# Adding secondary structure features
 - add a new feature type .yaml file in ./config/features
 - add the calculation in the ./features/node_features.py (compute_scalar_node_features)
 - register its dimension in the ./models/utils.py (get_input_dim)
 - remember to use your new feature configuration in the training arguments

# Adding TopoteinFeaturiser
 - inherit the ProteinFeaturiser in the ./features folder, override forward and __repr\__ method
 - add logic for attaching cells in the forward method
 - define cell types and features as well as their typings in the ./types.py
 - pass the cell type and features in the ./config/features/ca_sc_sse.yaml

# Adding my own model


# Launcher
pip install hydra-submitit-launcher
pip install submitit

# Scheduler
pip install flash lightning-flash torchmetrics>=1.2.0,2.0.0 pytorch-lightning>=2.0.7,<3.0.0
 jsonargparse-4.9.0 lightning-flash-0.8.1 pytorch-lightning-2.5.0.post0 torchmetrics-1.6.1

# test ckpt without training/finetune
 - python ./ProteinWorkshop/proteinworkshop/finetune.py encoder=tcpnet task=multilabel_graph_classification +aux_task=nn_structure_r3 dataset=go-mf features=ca_bb_sse  dataset.datamodule.dataset_fraction=0.01 trainer=gpu trainer.devices=1 logger=wandb +test=True trainer.max_epochs=0 name="tcp-mf-struct" ckpt_path=/home/drizer/PycharmProjects/Topotein/checkpoints/go/tcp/nn_structure_r3/tcp_mf_epoch_023.ckpt +no_training=True ++finetune.decoder.load_weights=True
 - python ./ProteinWorkshop/proteinworkshop/finetune.py encoder=gcpnet task=multilabel_graph_classification dataset=go-mf features=ca_bb  dataset.datamodul
e.dataset_fraction=0.01 trainer=gpu trainer.devices=1 logger=wandb +test=True trainer.max_epochs=0 name="gcp-mf-struct" ckpt_path=/home/drizer/PycharmProjects/Topotein/checkpoints/go/gcp/nn_structure_r3/gcp_mf_epoch_027.ckpt +no_training=True ++finetune.decoder.load_weights=True +aux_task=nn_structure_r3
