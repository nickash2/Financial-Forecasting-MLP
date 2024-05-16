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
- [ ] Pick which group of timeseries to use
- [ ] De-trend and de-seasonalize
## NN Modeling
- [ ] RNN vs MLP (**MLP recommended**)
- [ ] **WIP**

## Training
- [ ] Pick which strategy to use for training (general purpose vs new model for every new timeseries)
- [ ] Pick our train, test, (validation) split? 
- [ ] Use sMAPe for cross-validation
- [ ] Ensure proper regularisation


# Useful Resources (feel free to add more)
- https://machinelearningmastery.com/time-series-trends-in-python/ 
- https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/ (in Keras)
