# Web Analysis Demo, GCN PyTorch version

This code is utilized to do node-level and graph-level classification.

### Models

Use the simple 2-layer Graph Convolutional Neural Network to take implement the classification work.

### Requirements

```shell
python==3.9.7

pandas==1.3.5

matplotlib==3.5.0

networkx==2.6.3

torch==1.10.1

dgl-cuda11.3==0.7.2

```

Using following command:

```bash

$ pip install -r requirements.txt

```

### Train and visualization

**OpenFlight Dataset**

```bash

$ python src/GCN_torch.py

```

**Default adjustable parameters**

> epoches = 20
> hidden_size = 8
> country = "United States"
