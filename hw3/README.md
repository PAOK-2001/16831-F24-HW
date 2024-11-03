# HW3

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
./scripts/run_Q1.sh
./scripts/run_Q2.sh
```

2. For plotting use tensorboard to visualize runs

```
tensorboard --logdir data  --bind_all
```

Or use the provided script to visualize stylized plots included in homework PDF

```
python3 rob831/scripts/plot_q1.py
```

Do note that you will need to upgrade matplotlib to use the plotting scripts.

```
pip install matplotlib=='3.6.0'
```


