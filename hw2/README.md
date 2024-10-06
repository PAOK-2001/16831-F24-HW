# HW2

Pablo Ortega Kral (portegak)

1. For running all questions, use the included bash scripts

```
chmod +x scripts/*.sh
```

```
./scripts/run_all all
```

#Note: you can  run each individual question by especify the arguments in the aformetioned script.

```
./scripts/run_all q5
./scripts/run_all q7
./scripts/run_all q8
```

2. For plotting use tensorboard to visualize runs

```
tensorboard --logdir data  --bind_all
```

Or use the provided script to visualize stylized plots included in homework PDF

```
python scripts/get_graphs.py
```


