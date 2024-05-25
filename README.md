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
# To do
## Preprocessing
- [x] Pick which group of timeseries to use => Monthly timeseries of category `MICRO`
- [x] De-trend and de-seasonalize
## NN Modeling
- [x] RNN vs MLP (**MLP recommended**) => **MLP**
- [x] Research

## Training
- [x] Pick which strategy to use for training (general purpose vs **new model for every new timeseries**)
- [x] Pick our train, test, (validation) split **=> using last 18 points as test**
- [ ] Use sMAPe for cross-validation & ensure proper regularisation **=>** **Using Optuna**


# Useful Resources (feel free to add more)
- https://machinelearningmastery.com/time-series-trends-in-python/ 
- https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/ (in Keras)
