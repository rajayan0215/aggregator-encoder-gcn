class Param:
    cora_mixed = {
        "dataset": "cora",
        "sampling": "mixed",
        "num_classes": 7,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_folds": 20,
        "dim1": 512,
        "dim2": 128,
        "sampler1": "hybrid",
        "sampler2": "random",
        "sample1": 7,
        "sample2": 4,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    citeseer_mixed = {
        "dataset": "citeseer",
        "sampling": "mixed",
        "num_classes": 6,
        "num_nodes": 3312,
        "num_features": 3703,
        "num_folds": 20,
        "dim1": 512,
        "dim2": 128,
        "sampler1": "hybrid",
        "sampler2": "random",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    pubmed_mixed = {
        "dataset": "pubmed",
        "sampling": "mixed",
        "num_classes": 3,
        "num_nodes": 19717,
        "num_features": 500,
        "num_folds": 20,
        "dim1": 512,
        "dim2": 128,
        "sampler1": "hybrid",
        "sampler2": "random",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 1024
    }

    cora_priority = {
        "dataset": "cora",
        "sampling": "importance",
        "num_classes": 7,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "priority",
        "sampler2": "priority",
        "sample1": 7,
        "sample2": 4,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    citeseer_priority = {
        "dataset": "citeseer",
        "sampling": "importance",
        "num_classes": 6,
        "num_nodes": 3312,
        "num_features": 3703,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "priority",
        "sampler2": "priority",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    pubmed_priority = {
        "dataset": "pubmed",
        "sampling": "importance",
        "num_classes": 3,
        "num_nodes": 19717,
        "num_features": 500,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "priority",
        "sampler2": "priority",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 1024
    }

    cora_rand = {
        "dataset": "cora",
        "sampling": "random",
        "num_classes": 7,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "random",
        "sampler2": "random",
        "sample1": 7,
        "sample2": 4,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    citeseer_rand = {
        "dataset": "citeseer",
        "sampling": "random",
        "num_classes": 6,
        "num_nodes": 3312,
        "num_features": 3703,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "random",
        "sampler2": "random",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    pubmed_rand = {
        "dataset": "pubmed",
        "sampling": "random",
        "num_classes": 3,
        "num_nodes": 19717,
        "num_features": 500,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "random",
        "sampler2": "random",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 1024
    }

    cora_hybd = {
        "dataset": "cora",
        "sampling": "hybrid",
        "num_classes": 7,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "hybrid",
        "sampler2": "hybrid",
        "sample1": 7,
        "sample2": 4,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    citeseer_hybd = {
        "dataset": "citeseer",
        "sampling": "hybrid",
        "num_classes": 6,
        "num_nodes": 3312,
        "num_features": 3703,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "hybrid",
        "sampler2": "hybrid",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 256
    }

    pubmed_hybd = {
        "dataset": "pubmed",
        "sampling": "hybrid",
        "num_classes": 3,
        "num_nodes": 19717,
        "num_features": 500,
        "num_folds": 20,
        "dim1": 128,
        "dim2": 128,
        "sampler1": "hybrid",
        "sampler2": "hybrid",
        "sample1": 10,
        "sample2": 5,
        "learning_rate": 0.5,
        "lr_decay": 0.005,
        "batch_size": 1024
    }