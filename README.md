# Deep Learning Framework

We had to try a series of deep learning architectures in a class I took at NUS. So I created a simple framework to run Fully Connected Networks for my Deep Learning class without writing code. I wrote configurations to create and run a network, and even do hyper-paramter search.

It saved me a lot of time - with reusable code for plots like train and validation loss so I don't have to write all that stuff everytime!

## Example 

Sample configuration for a single network:

```json
{
      "data_path": "data/data_full3.zip",
      "model_name": "first-model-batch-",
      "structure": ["l_300_125", "d_0.3", "r", "l_125_125", "d_0.3",  "r", "l_125_150", "d_0.3", "r", "l_150_110", "d_0.3", "r", "l_110_75", "l_75_10", "s_0"],
      "lr": 3e-4,
      "epochs": 200,
      "input_dim": 50,
      "embedding_type": "glove-wiki-gigaword-50",
      "down_sample": {
        "drama": 0.0,
        "comedy": 0.0
      }
}
```

Sample configuration to run a series of architectures:

```json
{
    "configs": [
      {
        "data_path": "data/data_full3.zip",
        "model_name": "first-model-batch-base",
        "structure": ["l_50_125", "d_0.3", "r", "l_125_125", "d_0.3",  "r", "l_125_150", "d_0.3", "r", "l_150_110", "d_0.3", "r", "l_110_75", "l_75_10", "s_0"],
        "lr": 3e-4,
        "epochs": 200,
        "input_dim": 50,
        "batch_size": 16,
        "simple": true,
        "embedding_type": "glove-wiki-gigaword-50",
        "down_sample": {
          "drama": 0.0,
          "comedy": 0.0
        }
      },
      {
        "data_path": "data/data_full3.zip",
        "model_name": "first-model-batch-dropout-0",
        "structure": ["l_50_125", "r", "l_125_125",  "r", "l_125_150", "r", "l_150_110", "r", "l_110_75", "l_75_10", "s_0"],
        "lr": 3e-4,
        "epochs": 200,
        "input_dim": 50,
        "batch_size": 16,
        "simple": true,
        "embedding_type": "glove-wiki-gigaword-50",
        "down_sample": {
          "drama": 0.0,
          "comedy": 0.0
        }
      }
    ]
}
```
