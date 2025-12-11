# Credit Risk Probability Model for Alternative Data

This repository contains an end to end implementation of a credit risk probability model for Bati Bank.  
The bank is partnering with an eCommerce platform to offer a buy now pay later product based on alternative transaction data.

The goal is to transform customer behavior into a credit risk signal, then use that signal to score new customers and support lending decisions.

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── scoring.py
│   └── config.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
````

## Credit Scoring Business Understanding

### Basel II and the need for an interpretable model

Basel II encourages banks to use internal models for credit risk as long as these models are transparent, well governed and rigorously validated.
This means risk managers, auditors and supervisors must be able to understand how the model works, reproduce its results and check that it behaves sensibly on different portfolios.

For this project that implies:

* Every input feature must be clearly defined and traceable back to the original eCommerce data.
* The relationship between features and predicted risk should be understandable, not mysterious.
* The bank should be able to document how the model was built, which assumptions were made and how performance is monitored over time.

An interpretable and well documented model makes it easier to explain to regulators why a customer received a given score, to defend the model during reviews and to adjust it safely when new data becomes available.

### Why we use a proxy variable instead of a direct default label

The eCommerce dataset does not contain a direct credit outcome such as loan repaid or loan written off.
Supervised learning still requires a target, so we define a proxy variable for credit risk, for example by using customer Recency, Frequency and Monetary value to label less engaged or low value customers as higher risk.

This approach lets us:

* Train and compare models using a consistent good vs bad label.
* Start building risk tools before the bank has its own long history of BNPL defaults.
* Make use of behavioral patterns that are likely related to repayment ability and willingness.

However, it introduces several business risks:

* The model may learn to predict engagement or profitability instead of true default risk.
* Some customers can be misclassified as high risk based on behavior even though they would repay, which affects fairness and customer experience.
* When the bank starts using real BNPL performance data, the proxy based model will need to be checked and possibly recalibrated or replaced.

Because of this, the proxy label should be treated as a temporary stand in that helps the bank get started, not as a perfect definition of default.

### Trade offs between simple and complex models in a regulated setting

In practice, credit scoring has a strong tradition of using relatively simple and interpretable models, especially logistic regression and scorecards built from Weight of Evidence features.
These models:

* Have clear, monotonic relationships between features and risk.
* Are straightforward to document, explain to non technical stakeholders and implement in production systems.
* Tend to be stable over time and easier to recalibrate when the portfolio changes.

More complex models such as gradient boosting machines can capture non linear interactions and often deliver higher predictive power, which can improve risk ranking and expected profitability.
However, they:

* Are harder to interpret directly because the final prediction is the result of many trees and splits.
* Require extra tools for explanation, such as feature importance or SHAP values, which themselves add another layer that must be managed and validated.
* Can be more sensitive to data drift, making monitoring and governance more demanding.

For a bank operating under Basel II, a sensible strategy is to use a simple, interpretable model as the primary production model and treat more complex models as challengers or decision support tools.
This balances regulatory expectations about transparency and control with the desire to use advanced machine learning to improve risk prediction.

## How this section relates to Task 1

This Credit Scoring Business Understanding section summarizes:

* Why Basel II pushes us toward interpretable, well documented models.
* Why, in the absence of a direct default label, we construct a proxy risk variable and what risks this brings.
* The main trade offs between simple models such as logistic regression and more complex models such as gradient boosting in the context of credit risk modeling.


