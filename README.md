---
title: Short-term Cryptocurrency Price Prediction
authors: Nathan Waltz, Funso Oje, Hongjin Zhang
---

# Short-term Crypto-currency Price Prediction

We will be utilizing a variety of time series forecasting techniques and testing their efficacy on the Yahoo finance dataset [^yahoo-finance]. 

- [x] Create basic directory structure
- [x] Implement Baseline
- [ ] Implement Oracle
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
│   │   ├── ADA-USD.csv
│   │   └── BTC-USD.csv
│   └── plots
├── LICENSE
├── README.md
├── report
└── src
    ├── data_loader
    │   ├── load.py
    │   └── __pycache__
    │       └── load.cpython-39.pyc
    ├── engine
    │   └── data_parser.py
    ├── main.py
    ├── models
    │   ├── baseline.py
    │   ├── oracle.py
    │   └── __pycache__
    │       └── baseline.cpython-39.pyc
    ├── plotting_utils
    │   └── bokeh_utils.py
    └── utils
        └── utility_functions.py
```

[^yahoo-finance]: https://finance.yahoo.com/quote/BTC-USD/history/
