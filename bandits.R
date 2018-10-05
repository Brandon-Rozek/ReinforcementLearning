# Repeatedly face with a choice of k different options/actions

# After each choice, receive a numerical reward
# chosen from a stationary probability distribution
# depending on the action taken

# Maximize reward over 1000 time steps
rm(list=ls())

### VALUE ESTIMATORS
mean_estimator = function(previous_rewards, initial_reward) {
  if (length(previous_rewards) == 0) {
    return(initial_reward);
  }
  return (mean(previous_rewards)) # Replace with actual logic here
}
alpha_value_estimator = function(alpha, previous_rewards, initial_reward) {
  if (length(previous_rewards) == 0) {
    return(initial_reward);
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
random_bandits = function(num_bandits, MIN_REWARD, MAX_REWARD) {
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
  return(sum(sapply(rewards, sum))); # First sum for each bandit then combine those sums
}


### SIMULATION FUNCTIONS
simulate_run = function(NUM_TRIES, num_bandits, value_estimator, action_selector, MIN_REWARD, MAX_REWARD) {
  bandits = random_bandits(num_bandits, MIN_REWARD, MAX_REWARD);
  # Initialize reward list
  rewards = lapply(1:num_bandits, function(x) {return(numeric(0))})
  # Initialize value_estimates
  value_estimates = numeric(num_bandits);
  for (i in 1:NUM_TRIES) {
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

simulate_runs = function(NUM_SIMULATIONS, NUM_TRIES, num_bandits, value_estimator, action_selector, MIN_REWARD, MAX_REWARD, cluster) {
  sum_reward_simulation = function(i_simulation) {
    sum_rewards(simulate_run(NUM_TRIES, num_bandits, value_estimator, action_selector, MIN_REWARD, MAX_REWARD)$rewards)
  }
  if (missing(cluster)) {
    return(sapply(1:NUM_SIMULATIONS, sum_reward_simulation))
  } else {
    return(parSapply(cluster, 1:NUM_SIMULATIONS, sum_reward_simulation))
  }
}

calculateAlphaRewards = function(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, INITIAL_VALUE, MIN_REWARD, MAX_REWARD, cluster) {
  alphas = seq(0,1, 0.1);
  alphaRewards = lapply(alphas, function(alpha) {
    cbind(alpha, 
          simulate_runs(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, 
                        function(x) { alpha_value_estimator(alpha, x, INITIAL_VALUE) }, function(x) { greedy_epsilon_selector(.1, x) }
                        , MIN_REWARD, MAX_REWARD, cluster));
  }) %>% reduce(rbind);
  colnames(alphaRewards) = c("Alpha", "TotalRewardForSim");
  return(alphaRewards);
}

plotAlphaRewards = function(alphaRewards) {
  alphaRewards = as.data.frame(alphaRewards)
  p <- ggplot(alphaRewards, aes(x=as.factor(Alpha), y=TotalRewardForSim)) +
    geom_boxplot() + ylim(0, max(alphaRewards$TotalRewardForSim)) + 
    theme_bw() + ggtitle("Total Rewards by Alpha Level") + xlab("Alpha") + ylab("Total Rewards")
  print(p);
  return(p);
}

calculateEpsilonRewards = function(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, INITIAL_VALUE, MIN_REWARD, MAX_REWARD, cluster) {
  epsilons = seq(0, 1, 0.1)
  epsilonRewards = lapply(epsilons, function(epsilon) {
    cbind(epsilon,
          simulate_runs(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS,
                        function(x) { mean_estimator(x, INITIAL_VALUE) } , function(x) { greedy_epsilon_selector(epsilon, x)}, 
                        MIN_REWARD, MAX_REWARD, cluster))
  }) %>% reduce(rbind);
  colnames(epsilonRewards) = c("Epsilon", "TotalRewardForSim");
  return(epsilonRewards);
}

plotEpsilonRewards = function(epsilonRewards) {
  epsilonRewards = as.data.frame(epsilonRewards);
  p <- ggplot(epsilonRewards, aes(x = as.factor(Epsilon), y = TotalRewardForSim)) +
    geom_boxplot() + ylim(0, max(epsilonRewards$TotalRewardForSim)) +
    theme_bw() + ggtitle("Total Rewards by Epsilon Level") + xlab("Epsilon") + ylab("Total Rewards") 
  print(p);
  return(p);
}

calculateInitialEstimates = function(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, MIN_REWARD, MAX_REWARD, cluster) {
  initialEstimates = seq(MIN_REWARD / 2, MAX_REWARD * 2, length.out = 10)
  initialRewards = lapply(initialEstimates, function(initial) {
    cbind(initial,
          simulate_runs(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS,
                        function(x) { mean_estimator(x, initial)}, greedy_selector
                        , MIN_REWARD, MAX_REWARD, cluster))
  }) %>% reduce(rbind);
  colnames(initialRewards) = c("InitialReward", "TotalRewardForSim");
  return(initialRewards);
}

plotInitialEstimates = function(initialRewards) {
  initialRewards = as.data.frame(initialRewards)
  p <- ggplot(initialRewards, aes(x = as.factor(InitialReward), y = TotalRewardForSim)) +
          geom_boxplot() + ylim(0, max(initialRewards$TotalRewardForSim)) + xlab("Initial Reward") + ylab("Total Rewards") +
          theme_bw() + ggtitle("Rewards by Initial Reward Chosen")
  print(p);
  return(p);
}

calcluateNumberBanditRewards = function(NUM_SIMULATIONS, NUM_TRIALS, INITIAL_VALUE, MIN_REWARD, MAX_REWARD, cluster) {
  numberBandits = seq(NUM_TRIALS / 2, NUM_TRIALS * 2, length.out = 10);
  numberBanditsRewards = lapply(numberBandits, function(num_bandit) {
    cbind(num_bandit,
          simulate_runs(NUM_SIMULATIONS, NUM_TRIALS, num_bandit,
                        function(x) { mean_estimator(x, INITIAL_VALUE)}, greedy_selector, MIN_REWARD, MAX_REWARD, cluster))
  }) %>% reduce(rbind);
  colnames(numberBanditsRewards) = c("NumBandits", "TotalRewardForSim");
  return(numberBanditsRewards);
}

plotNumberBanditRewards = function(numberBanditsRewards) {
  numberBanditsRewards = as.data.frame(numberBanditsRewards)
  p <- ggplot(numberBanditsRewards, aes(x = as.factor(NumBandits), y = TotalRewardForSim)) +
    geom_boxplot() + ylim(0, max(numberBanditsRewards$TotalRewardForSim)) + xlab("Num Bandits") + ylab("Total Rewards") +
    theme_bw() + ggtitle("Rewards by Number of Bandits")
  print(p);
  return(p);
}

generateDataset = function(cluster) {
  NUM_TRIALS = 2; # This gives us different parameters to try out
  NUM_RUNS = 200; # This gives us a way to average each parameter set
  NUM_ITER_PER_RUN = 1000; # Goal is to maximize 1000 runs
  min_rewards = runif(NUM_TRIALS, 0, 500);
  max_rewards = min_rewards + 500;
  initial_values = runif(NUM_TRIALS, 500, 1000);
  num_bandits = sample(5:5000, NUM_TRIALS, replace = T); 
  epsilons = runif(NUM_TRIALS, 0, 1);
  rewards = mapply(function(min_reward, max_reward, initial_value, num_bandit, epsilon, i) {
    if (((i - 1) / NUM_TRIALS) * 100 %% 10 == 0 ) { cat(paste(((i - 1) / NUM_TRIALS) * 100, "% Complete...", sep = "")) }
    result = simulate_runs(NUM_RUNS, NUM_ITER_PER_RUN, num_bandit,
                           function(x) { mean_estimator(x, initial_value) }, function(x) { greedy_epsilon_selector(epsilon, x) }, 
                           min_reward, max_reward, cluster);
    return(cbind(min_reward, max_reward, initial_value, num_bandit, epsilon, num_bandit / NUM_ITER_PER_RUN, result))
  }, min_rewards, max_rewards, initial_values, num_bandits, epsilons, 1:NUM_TRIALS) %>% reduce(rbind);
  colnames(rewards) = c("min_reward", "max_reward", "initial_value", "num_bandits", "epsilon", "bandit_iter_ratio", "total_reward");
  return(rewards)
}
################# MAIN
library(parallel)
library(purrr)
library(ggplot2)



plotAll = function() {
  ##### MAIN BELOW
  MIN_REWARD = 500
  MAX_REWARD = 1000
  AVG_REWARD = (MAX_REWARD-MIN_REWARD)/2 + MIN_REWARD
  INITIAL_VALUE = AVG_REWARD * .5;
  NUM_SIMULATIONS = 1000;
  NUM_TRIALS = 100
  NUM_BANDITS = 5;
  
  no_cores = detectCores() - 1;
  cl = makeCluster(no_cores, type = "FORK");
  
  cat("Calculating Alpha Rewards...\n");
  alphaRewards = calculateAlphaRewards(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, INITIAL_VALUE, MIN_REWARD, MAX_REWARD, cl);
  plotAlphaRewards(alphaRewards);
  
  cat("Calculating Epsilon Rewards...\n");
  epsilonRewards = calculateEpsilonRewards(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, INITIAL_VALUE, MIN_REWARD, MAX_REWARD, cl);
  plotEpsilonRewards(epsilonRewards);
  
  cat("Calculating Initial Estimates Rewards...\n");
  initialRewards = calculateInitialEstimates(NUM_SIMULATIONS, NUM_TRIALS, NUM_BANDITS, MIN_REWARD, MAX_REWARD, cl);
  plotInitialEstimates(initialRewards);
  
  cat("Calculating Number Bandits Rewards...\n");
  banditRewards = calcluateNumberBanditRewards(NUM_SIMULATIONS, NUM_TRIALS, INITIAL_VALUE, MIN_REWARD, MAX_REWARD, cl);
  plotNumberBanditRewards(banditRewards);
  
  cat("\n\nFinished.\n");
  stopCluster(cl);
}

