---
title: Short-term Cryptocurrency Price Prediction
authors: Nathan Waltz, Funso Oje, Hongjin Zhang
---

# Short-term Crypto-currency Price Prediction

We will be utilizing a variety of time series forecasting techniques and testing their efficacy on the Yahoo finance dataset [^yahoo-finance]. 

- [x] Create basic directory structure
- [X] Implement Baseline
- [X] Implement Oracle
- [X] Added NNI for hyperparameter tuning
- [X] Create custom model(s) for data
- [X] Develop loss function(s) for comparing performance
- [ ] Figure out a way to measure power consumption and execution time
- [ ] Train on all datasets
- [ ] Train on more datasets

Our project is structured as follows

```
.
├── LICENSE
├── README.md
├── assets
│   └── datasets
│       ├── BTC-USD.csv
│       └── ETH-USD.csv
├── environment.yml
├── main.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── data_loader
    │   ├── __init__.py
    │   └── load.py
    ├── models
    │   ├── __init__.py
    │   ├── baseline
    │   │   └── baseline.py
    │   ├── custom
    │   │   └── custom.py
    │   ├── loss.py
    │   ├── oracle
    │   │   ├── data
    │   │   │   └── oracle_best.pth
    │   │   └── oracle.py
    │   └── utils.py
    └── tuning
        ├── __init__.py
        ├── custom_hyp_conf.yml
        ├── custom_trial.py
        ├── oracle_hyp_conf.yml
        ├── oracle_trial.py
        ├── search_space_custom.json
        └── search_space_oracle.json
```

[^yahoo-finance]: https://finance.yahoo.com/quote/BTC-USD/history/
