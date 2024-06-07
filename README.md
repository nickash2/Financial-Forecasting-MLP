# Neural Networks Project
## Topic: Financial timeseries forecasting
Our project will focus on employing an MLP to predict financial timeseries data, aiming to leverage their sequential learning capabilities for "accurate" forecasting of stock prices and market trends.

> [!IMPORTANT]
> Please use a different branch before committing to the main branch!
> 
> **Commit everything to the dev branch so we can all approve it to the main when appropriate.**


# Environment Setup

To make sure we are all on the same latest versions, make an environment like so:

```bash
python3 -m venv project
source project/bin/activate
pip install -r requirements.txt
```
# Optuna Dashboard
This will open the dashboard on your local machine at
http://127.0.0.1:8080/.
```bash
optuna-dashboard sqlite:///data/tuning.db
```

# To do
## Preprocessing
- [x] Pick which group of timeseries to use => Monthly timeseries of category `MICRO`
- [x] De-trend and de-seasonalize

## NN Modeling
- [x] RNN vs MLP (**MLP recommended**) => **MLP**
- [x] Research

## Training
- [x] Pick which strategy to use for training (**general purpose** vs new model for every new timeseries)
- [x] Pick our train, test, (validation) split **=> using an 80-20 split (with validation)**
- [x] Use sMAPe for cross-validation & ensure proper regularisation **=>** **Using Optuna w/ Habrok**
- [x] Split data appropriately **=> using blocked cross validation**
- [x] Iteravely train the model and tune it w/ Optuna
- [x] Plot training losses, optimization history etc.

## Predicting
- [ ] Predict and compare accuracy with the test set using sMAPe
- [ ] Retrend the data for visual inspection and plotting on a chosen timeseries


## Report Checklist
- [ ] Read provided reports to create a structure of our report
- [ ] Divide sections per teammate
- [ ] Do a cross-check on the report



# Report Resources
## Optuna
- [Tree-Structured Parzen Estimator.](https://ar5iv.labs.arxiv.org/html/2304.11127)
- [Algorithms for Hyper-Parameter Optimization.](https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
- [Information on Optuna's Hyperband Pruner that we use.](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf)
- [Optuna Dashboard Generated Plots](#optuna-dashboard)


# Useful Resources (feel free to add more)
- https://machinelearningmastery.com/time-series-trends-in-python/ 
- https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/ (in Keras)
- https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
