####################################################
# RESULTS:
# Linear Weights: [-0.06212267]
# Linear BIAS: 4.252692612978406
# 
# Logistic Weights: [[0.06209602]]
# Logistic BIAS: [-4.23788707]
# 
# Probability at HR 60: 0.375
####################################################


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

## INPUT DATA
heart_rates = np.array([50, 70, 90])
# (#No's)/(#Yes's)
odds = [3, 1, .25]
logits =  np.log(odds)

## Create linear regressor
LR = LinearRegression()
LR.fit(heart_rates.reshape(-1,1), logits)
print("Linear Weights:",LR.coef_)
print("Linear BIAS:",LR.intercept_)

## PREDICT for HR=60

## Example - plot the data / best fit line
## How would we plot the line transformed back into probability space (i.e. the sigmoid one)?
plt.scatter(heart_rates, logits)
x = np.linspace([0], [150]) # create a list of evenly spaced values between 0 and 150 - your x values
y = LR.predict(x) # this gives you the logits (would need to run this through a sigmoid to get the probability)
plt.plot(x,y)
plt.show()

# Same for logistic regression, but note it expects the original data (not the logit version)
all_heart_rates = np.array([50, 50, 50, 50, 70, 70, 90, 90, 90, 90, 90]).reshape(-1,1)
heart_attacks = [True,False,False,False,False,True,True,True,False,True,True]
LOR = LogisticRegression(random_state=0, solver='lbfgs')
LOR.fit(all_heart_rates,heart_attacks)
print("\nLogistic Weights:",LOR.coef_)
print("Logistic BIAS:",LOR.intercept_)
probs = np.array(LOR.predict_proba(all_heart_rates))
plt.plot(all_heart_rates, probs[:,1])
probability = LOR.predict_proba([[60]])[0][1]
# round to 3 sig figs
print("\nProbability at HR 60:",round(probability,3))
plt.scatter(60, probability)
plt.show()