---
title: Short-term Cryptocurrency Price Prediction
authors: Nathan Waltz, Funso Oje, Hongjin Zhang
---

# Short-term Crypto-currency Price Prediction

We will be utilizing a variety of time series forecasting techniques and testing their efficacy on the Yahoo finance dataset [^yahoo-finance]. 

- [x] Create basic directory structure
- [ ] Implement Baseline
- [X] Implement Oracle
- [X] Added NNI for hyperparameter tuning
- [ ] Improve Baseline fitting?
- [ ] Create custom model(s) for data
- [ ] Develop loss function(s) for comparing performance
- [ ] Figure out a way to measure power consumption and execution time
- [ ] Train on all datasets
- [ ] Train on more datasets

Our project is structured as follows

```
.
├── assets
│   ├── datasets
│   │   ├── BTC-USD.csv
│   │   └── ETH-USD.csv
│   └── plots
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── src
    ├── data_loader
    │   ├── __init__.py
    │   ├── load.py
    │   └── __pycache__
    │       ├── __init__.cpython-38.pyc
    │       ├── __init__.cpython-39.pyc
    │       ├── load.cpython-38.pyc
    │       └── load.cpython-39.pyc
    ├── __init__.py
    ├── models
    │   ├── baseline
    │   │   └── baseline.py
    │   ├── __init__.py
    │   ├── oracle
    │   │   ├── data
    │   │   │   └── oracle_best.pth
    │   │   ├── oracle.py
    │   │   └── __pycache__
    │   │       ├── oracle.cpython-38.pyc
    │   │       └── oracle.cpython-39.pyc
    │   └── __pycache__
    │       ├── __init__.cpython-38.pyc
    │       └── __init__.cpython-39.pyc
    ├── __pycache__
    │   ├── __init__.cpython-38.pyc
    │   └── __init__.cpython-39.pyc
    └── tuning
        ├── __init__.py
        ├── oracle_hyp_conf.yml
        ├── oracle_trial.py
        ├── __pycache__
        │   ├── __init__.cpython-39.pyc
        │   └── oracle_trial.cpython-39.pyc
        └── search_space.json
```

[^yahoo-finance]: https://finance.yahoo.com/quote/BTC-USD/history/
