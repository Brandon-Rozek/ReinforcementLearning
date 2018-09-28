# Repeatedly face with a choice of k different options/actions

# After each choice, receive a numerical reward
# chosen from a stationary probability distribution
# depending on the action taken

# Maximize reward over 1000 time steps
rm(list=ls())

MIN_REWARD = 500
MAX_REWARD = 1000
AVG_REWARD = (MAX_REWARD-MIN_REWARD)/2 + MIN_REWARD


### VALUE ESTIMATORS
mean_estimator = function(previous_rewards) {
  if (length(previous_rewards) == 0) {
    return(AVG_REWARD*.5); # Initial value goes here
  }
  return (mean(previous_rewards)) # Replace with actual logic here
}
alpha_value_estimator = function(alpha, previous_rewards) {
  if (length(previous_rewards) == 0) {
    return(AVG_REWARD*.5); # Initial value goes here
  }
  triesWithBandits = length(previous_rewards);
  reward_indices = 1:triesWithBandits
  return(sum(mapply(
                function(reward, num_tries, index) { return (alpha * (1 - alpha)^(num_tries - index) * reward)}, 
                previous_rewards,
                triesWithBandits,
                reward_indices)))
}



### ACTION SELECTORS
greedy_selector = function(value_estimates) {
  return(which.max(value_estimates));
}
greedy_epsilon_selector = function(epsilon, value_estimates) {
  if (runif(1, 1, 100) < epsilon * 100) {
    return(sample(1:length(value_estimates), 1))
  } else {
    return(greedy_selector(value_estimates))
  }
}


### BANDIT CREATION
create_bandit = function(mu_reward, std_reward) {
  return(list(mu = mu_reward, std = std_reward))
}
random_bandits = function(num_bandits) {
  return(t(mapply(create_bandit, 
                runif(num_bandits, MIN_REWARD, MAX_REWARD),
                rexp(num_bandits, 1))))
}

### REWARD FUNCTIONS
get_reward = function(bandits, bandit_number) {
  if (bandit_number > nrow(bandits)) {
    errorMessage = paste("Cannot select bandit number",
                         bandit_number,
                         "from a possible",
                         nrow(bandits),
                         "bandits.");
    stop(errorMessage)
  }
  
  # Assume rewards are given from a normal pdf
  bandit_chosen = bandits[bandit_number, ];
  reward = rnorm(1, bandit_chosen$mu, bandit_chosen$std);
  return(reward)
}
sum_rewards = function(rewards) {
  return(sum(sapply(rewards, sum)));
}


### SIMULATION FUNCTIONS
simulate_run = function(num_tries, num_bandits, value_estimator, action_selector) {
  bandits = random_bandits(num_bandits);
  # Initialize reward list
  rewards = lapply(1:num_bandits, function(x) {return(numeric(0))})
  # Initialize value_estimates
  value_estimates = numeric(num_bandits);
  for (i in 1:num_tries) {
    value_estimates = sapply(rewards, value_estimator);
    selected_bandit = action_selector(value_estimates);
    bandit_reward = get_reward(bandits, selected_bandit);
    rewards[[selected_bandit]] = c(rewards[[selected_bandit]], bandit_reward);
  }
  return(list(
    bandits = bandits,
    rewards = rewards,
    value_estimates = value_estimates
  ))
}
simulate_runs = function(num_simulations, num_tries, num_bandits, value_estimator, action_selector) {
  total_rewards = numeric(num_simulations);
  for (i in 1:num_simulations) {
    total_rewards[i] = sum_rewards(simulate_run(num_tries, num_bandits, value_estimator, action_selector)$rewards);
  }
  return(list(
    mu = mean(total_rewards),
    std = sd(total_rewards),
    rewards = total_rewards
  ))
}


# Mean estimators, greedy selector
#meangreedy = simulate_runs(1000, 100, 5, mean_estimator, greedy_selector);

epsilons = seq(0,.5,.05)
mu_vector = numeric(0);
md_vector = numeric(0);
sd_vector = numeric(0);
stephen <- data.frame(epsilon=numeric(0), value=numeric(0))
for (epsilon in epsilons) {
  cat("epsilon =",epsilon,"...\n")
  current_run = simulate_runs(1000, 100, 5, mean_estimator, function(x) { greedy_epsilon_selector(epsilon, x) });
  mu_vector = c(mu_vector, current_run$mu);
  #md_vector = c(md_vector, median(current_run$rewards))
  sd_vector = c(sd_vector, current_run$std);
    stephen <- rbind(stephen,cbind(epsilon, current_run$rewards))
}


library(ggplot2)
p <- ggplot(data.frame(epsilons, mu_vector), aes(x = epsilons, y = mu_vector)) + geom_point() + ggtitle("Mean Reward") + theme_bw() + ylim(0,max(mu_vector))
#print(p)
X11()
#ggplot(data.frame(epsilons, md_vector), aes(x = epsilons, y = md_vector)) + geom_point() + ggtitle("Median Reward") + theme_bw()
p2 <- ggplot(data.frame(epsilons, sd_vector), aes(x = epsilons, y = sd_vector)) + geom_point()  + ggtitle("Standard Deviation of Rewards") + theme_bw() + ylim(0,max(sd_vector))
#print(p2)

colnames(stephen) <- c("epsilon","value")
p3 <- ggplot(stephen, aes(x=as.factor(epsilon), y=value)) + geom_boxplot()
print(p3)
