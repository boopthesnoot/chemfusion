# chemfusion: fusing graph and text representations of molecules for property prediction

used parts of the code from:
https://github.com/MolecularAI/Chemformer

usage: specify parameters, datasets and models used in `graph/config.yaml` and run `python graph/main.py` with 
flags `--vocab_path` to specify the vocabulary  for the NLP part and `--model_path` to specify path to the 
pretrained chemformer model (https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq)

Results on freesolv:

| Regression Model                    | Fusion                             | Dataset  | RMSE Mean (test) | RMSE Std (test) | RMSE (test)    | R2 (test)      | Flags         |
|-------------------------------------|------------------------------------|----------|------------------|-----------------|----------------|----------------|---------------|
| Transformer + GCN wo Reconstruction | Concat -> 2 dense layers with ReLU | freesolv | 1.450            | 0.221           | 1.450 +- 0.221 | 0.901 +- 0.030 | run.n_runs=10 |
| Transformer + GCN w/ Reconstruction | Concat -> 2 dense layers with ReLU | freesolv | 1.451            | 0.103           | 1.451 +- 0.103 | 0.903 +- 0.014 | run.n_runs=10 |
| GCN wo Reconstruction               |                                    | freesolv | 1.803            | 0.060           | 1.803 +- 0.060 | 0.849 +- 0.010 | run.n_runs=10 |
| GCN w/ Reconstruction               |                                    | freesolv | 1.904            | 0.044           | 1.904 +- 0.044 | 0.833 +- 0.008 | run.n_runs=10 |
| Transformer wo Reconstruction       |                                    | freesolv | 2.071            | 0.291           | 2.071 +- 0.291 | 0.796 +- 0.057 | run.n_runs=10 |
| Transformer w/ Reconstruction       |                                    | freesolv | 2.104            | 0.381           | 2.104 +- 0.381 | 0.787 +- 0.084 | run.n_runs=10 |