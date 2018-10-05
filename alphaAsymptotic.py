
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_bandits(alpha=.5,num_trials=1000):

    stdevs = [0]*num_trials
    final_guesses = [0]*num_trials
    for trial in range(num_trials):
        print("Currently on trial " + str(trial) + "...")
        one_bandits_rewards = np.random.exponential(.5,10)
        sample_mean = pd.Series(one_bandits_rewards).mean()
        #print("sample mean was: {:.4f}".format(sample_mean))

        prediction = [0] * len(one_bandits_rewards)
        prediction[0] = one_bandits_rewards[0]
        for num_iter in range(1,len(one_bandits_rewards)):
            prediction[num_iter] = prediction[num_iter - 1] + \
                alpha/num_iter * (one_bandits_rewards[num_iter] - prediction[num_iter - 1])

        #print("Standard dev: {:.4f}".format(pd.Series(prediction).std()))
        stdevs[trial] = pd.Series(prediction).std()
        final_guesses[trial] = prediction[-1]


    return pd.Series(stdevs).std(), pd.Series(final_guesses).mean()


alphas = np.arange(0,20,.5)
stdevs = np.empty(len(alphas))
final_guesses = np.empty(len(alphas))
for i,alpha in enumerate(alphas):
    stdevs[i], final_guesses[i] = run_bandits(alpha)
    
plt.clf()
#plt.scatter(alphas,stdevs)
plt.scatter(alphas,final_guesses)
plt.axhline(.5,linestyle='dotted')
plt.show()

