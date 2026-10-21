| dataset | model | test acc | train-test gap | best attack | AUC | TPR@0.1%FPR | TPR@1%FPR | risk |
|---|---|---|---|---|---|---|---|---|
| digits | LogReg (C=0.05) | 0.952 | +0.028 | lira | 0.615 | 0.008 | 0.033 | **moderate** |
| digits | RandomForest | 0.973 | +0.027 | lira | 0.849 | 0.008 | 0.039 | **severe** |
| digits | GradBoost | 0.949 | +0.051 | lira | 0.796 | 0.079 | 0.186 | **high** |
| digits | MLP (no reg.) | 0.973 | +0.027 | lira | 0.657 | 0.047 | 0.081 | **high** |
| digits | MLP (L2 + early stop) | 0.937 | +0.040 | lira | 0.538 | 0.000 | 0.016 | **low** |
| breast_cancer | LogReg (C=0.05) | 0.968 | +0.011 | entropy | 0.514 | 0.004 | 0.021 | **moderate** |
| breast_cancer | RandomForest | 0.947 | +0.053 | lira | 0.662 | 0.004 | 0.035 | **high** |
| breast_cancer | GradBoost | 0.947 | +0.053 | lira | 0.621 | 0.014 | 0.081 | **high** |
| breast_cancer | MLP (no reg.) | 0.982 | +0.018 | lira | 0.567 | 0.014 | 0.025 | **moderate** |
| breast_cancer | MLP (L2 + early stop) | 0.933 | +0.007 | lira | 0.515 | 0.007 | 0.018 | **low** |
| synthetic-4class | LogReg (C=0.05) | 0.653 | +0.010 | lira | 0.545 | 0.009 | 0.019 | **low** |
| synthetic-4class | RandomForest | 0.729 | +0.271 | lira | 1.000 | 0.971 | 0.989 | **severe** |
| synthetic-4class | GradBoost | 0.739 | +0.206 | lira | 0.852 | 0.088 | 0.237 | **severe** |
| synthetic-4class | MLP (no reg.) | 0.686 | +0.314 | lira | 0.897 | 0.110 | 0.391 | **severe** |
| synthetic-4class | MLP (L2 + early stop) | 0.685 | +0.187 | lira | 0.753 | 0.058 | 0.145 | **high** |
