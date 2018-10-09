__precompile__()

## Helper functions
function randomab(min_num :: Float64, max_num :: Float64)
    return rand() * (max_num - min_num + 1.0) + min_num
end



## Bandit Creation Functions
struct Bandit
	mu :: Float64 
	std :: Float64
end

using Distributions
# Max number of bandits is 65535
function random_bandits(num_bandits :: UInt16, min_reward :: Float64, max_reward :: Float64)
	bandits = Array{Bandit, 2}(undef, num_bandits, 1)
	sd_dist = Exponential(1)
	for ii = 1:num_bandits
		bandits[ii] = Bandit(randomab(min_reward, max_reward), rand(sd_dist))
	end
	return bandits
end


## Value Estimators
function mean_estimator(previous_rewards :: Array{Float64, 1}, initial_reward :: Float64)
	if length(previous_rewards) == 0
		return initial_reward
	end
	return mean(previous_rewards)
end

function alpha_value_estimator(alpha :: Float32, previous_rewards :: Array{Float64, 1}, initial_reward :: Float64)
	triesWithBandits = length(previous_rewards)
	if triesWithBandits == 0
		return initial_reward
	end

	result = 0.0
	for reward_index = 1:triesWithBandits
		result += alpha * (1.0 - alpha)^(triesWithBandits - reward_index) * previous_rewards[reward_index]
	end
	return result
end

## Action Selectors
function greedy_selector(value_estimates :: Array{Float64})
	_, index = findmax(value_estimates)
	return convert(UInt32, index)
end

function greedy_epsilon_selector(epsilon :: Float32, value_estimates :: Array{Float64})
	if randomab(1.0, 100.0) < epsilon * 100
		return convert(UInt32, rand(1:length(value_estimates)))
	end
	return greedy_selector(value_estimates)
end


function get_reward(bandits :: Array{Bandit}, bandit_num :: UInt32)
	bandit_length = length(bandits)
	if bandit_num > bandit_length
		error("Cannot selected bandit number $bandit_num from $bandit_length")
	end
	# Assume rewards come from a gaussian distribution
	d = Normal(bandits[bandit_num].mu, bandits[bandit_num].std)
	return rand(d)
end
mutable struct BanditRewards
	rewards :: Array{Float64, 1}
end

function sum_rewards(rewards :: Array{BanditRewards})
	sum = 0
	for bandit = 1:length(rewards)
		bandit_rewards = rewards[bandit].rewards
		for reward_index = 1:length(bandit_rewards)
			sum += bandit_rewards[reward_index]
		end
	end
	return sum
end

## Simulation Functions
function simulate_run(num_tries :: UInt64, num_bandits :: UInt16, value_estimator :: Function, action_selector :: Function, min_reward :: Float64, max_reward :: Float64)
	bandits = random_bandits(num_bandits, min_reward, max_reward)
	# Allocate num_bandits empty arrays
	rewards = Array{BanditRewards,2}(undef, num_bandits, 1) 
	for i_bandit = 1:num_bandits
		rewards[i_bandit] = BanditRewards([])
	end
	# Above loop needed to have rewards initialized
	value_estimates = zeros(num_bandits)
	for i_try = 1:num_tries
		for i_bandit = 1:num_bandits
			value_estimates[i_bandit] = value_estimator(rewards[i_bandit].rewards)
		end
		selected_bandit = action_selector(value_estimates)
		bandit_reward = get_reward(bandits, selected_bandit)
		append!(rewards[selected_bandit].rewards, bandit_reward)
	end
	return rewards
end

function simulate_runs(num_simulations :: UInt64, num_tries :: UInt64, num_bandits :: UInt16, value_estimator :: Function, action_selector :: Function, min_reward :: Float64, max_reward :: Float64)
	rewards = zeros(num_simulations)
	for ii = 1:num_simulations
		rewards[ii] = sum_rewards(simulate_run(num_tries, num_bandits, value_estimator, action_selector, min_reward, max_reward))
	end
	return rewards
end

using Distributed
@everywhere using SharedArrays
function parallel_simulate_runs(num_simulations :: UInt64, num_tries :: UInt64, num_bandits :: UInt16, value_estimator :: Function, action_selector :: Function, min_reward :: Float64, max_reward :: Float64)
        int_num_simulations = convert(Int64, num_simulations)
        rewards = SharedArray{Float64, 1}(int_num_simulations)
        @sync @distributed for ii = 1:int_num_simulations
                rewards[ii] = sum_rewards(simulate_run(num_tries, num_bandits, value_estimator, action_selector, min_reward, max_reward))
        end
        return rewards
end


## One dimensional parameter sweeps
function calculateAlphaRewards(num_simulations :: UInt64, num_trials :: UInt64, num_bandits :: UInt16, initial_value :: Float64, min_reward :: Float64, max_reward :: Float64)
	alphas = collect(convert(Float32, 0.0):convert(Float32, 0.1):convert(Float32,1))
	alphaRewards = zeros(0, 2)
	for i_alpha = 1:length(alphas)
		alpha = alphas[i_alpha]
		i_alphas = fill(alpha, (num_simulations, 1))
		alphaRewards = vcat(alphaRewards, hcat(i_alphas, parallel_simulate_runs(num_simulations, num_trials, num_bandits, (x) -> alpha_value_estimator(alpha, x, initial_value), (x) -> greedy_epsilon_selector(convert(Float32, .1), x), min_reward, max_reward)))
	end
	return alphaRewards
end
function calculateEpsilon(num_simulations :: UInt64, num_trials :: UInt64, num_bandits :: UInt16, initial_value :: Float64, min_reward :: Float64, max_reward :: Float64)
        epsilons = collect(convert(Float32, 0.0):convert(Float32, 0.1):convert(Float32,1))
        epsilonRewards = zeros(0, 2)
        for i_epsilon = 1:length(epsilons)
                epsilon = epsilons[i_epsilon]
                i_epsilons = fill(epsilon, (num_simulations, 1))
                epsilonRewards = vcat(epsilonRewards, hcat(i_epsilons, parallel_simulate_runs(num_simulations, num_trials, num_bandits, (x) -> mean_estimator(x, initial_value), (x) -> greedy_epsilon_selector(epsilon, x), min_reward, max_reward)))
        end
        return epsilonRewards
end
function calculateInitialEstimates(num_simulations :: UInt64, num_trials :: UInt64, num_bandits :: UInt16, min_reward :: Float64, max_reward :: Float64)
        initialEstimates = collect(linspace(min_reward / 2, max_reward / 2, 10))
        initialEstimatesRewards = zeros(0, 2)
        for i_initial = 1:length(initialEstimates)
                initial = initialEstimates[i_initial]
                i_initials = fill(initial, (num_simulations, 1))
                initialEstimatesRewards = vcat(initialEstimatesRewards, hcat(i_initials, parallel_simulate_runs(num_simulations, num_trials, num_bandits, (x) -> mean_estimator(x, initial), (x) -> greedy_epsilon_selector(convert(Float32, .1), x), min_reward, max_reward)))
        end
        return initialEstimatesRewards
end
function calculateNumBanditRewards(num_simulations :: UInt64, num_trials :: UInt64, initial_estimate :: Float64, min_reward :: Float64, max_reward :: Float64)
        number_bandits = collect(linspace(convert(UInt16, floor(num_trials / 2)), convert(UInt16, floor(num_trials * 2)), 10))
        numBanditRewards = zeros(0, 2)
        for i_bandit = 1:length(number_bandits)
                num_bandit = number_bandits[i_bandit]
                i_banditss = fill(num_bandit, (num_simulations, 1))
                numBanditRewards = vcat(numBanditRewards, hcat(i_bandits, parallel_simulate_runs(num_simulations, num_trials, num_bandit, (x) -> mean_estimator(x, initial_estimate), (x) -> greedy_epsilon_selector(convert(Float32, .1), x), min_reward, max_reward)))
        end
        return numBanditRewards
end

function generateDatasetNoAlpha(num_permutations :: UInt64, num_simulations :: UInt64, num_trials :: UInt64)
	min_rewards = Uniform(0, 500)
	initial_values = Uniform(0, 1000)
	num_bandits = DiscreteUniform(5, 5000)
	epsilons = Uniform(0, 1)
	rewards = zeros(0, 6)
	for i_perm = 1:num_permutations
		min_reward = rand(min_rewards)
		max_reward = min_reward + 500
		initial_value = rand(initial_values)
		num_bandit = rand(num_bandits)
		epsilon = rand(epsilons)
		rewards = vcat(rewards, hcat(fill(min_reward, (num_simulations, 1)), fill(max_reward, (num_simulations, 1)), fill(initial_value, (num_simulations, 1)), 
						fill(num_bandit, (num_simulations, 1)), fill(epsilon, (num_simulations, 1)), 
						parallel_simulate_runs(num_simulations, num_trials, convert(UInt16, num_bandit), (x) -> mean_estimator(x, initial_value),
										(x) -> greedy_epsilon_selector(convert(Float32, epsilon), x), min_reward, max_reward)))
	end
	return rewards
end
using Base.Threads
function generateDataset(num_permutations :: UInt64, num_simulations :: UInt64, num_trials :: UInt64)
        min_rewards = Uniform(0, 500)
        initial_values = Uniform(0, 1000)
        num_bandits = DiscreteUniform(5, 5000)
        epsilons = Uniform(0, 1)
	alphas = Uniform(0, 1)
        rewards = zeros(0, 7)
        @threads for i_perm = 1:num_permutations
                min_reward = rand(min_rewards)
                max_reward = min_reward + 500
                initial_value = rand(initial_values)
                num_bandit = rand(num_bandits)
                epsilon = rand(epsilons)
		alpha = rand(alphas)
                rewards = vcat(rewards, hcat(fill(min_reward, (num_simulations, 1)), fill(max_reward, (num_simulations, 1)), fill(initial_value, (num_simulations, 1)),
                                                fill(num_bandit, (num_simulations, 1)), fill(epsilon, (num_simulations, 1)), fill(alpha, (num_simulations, 1)), 
                                                parallel_simulate_runs(num_simulations, num_trials, convert(UInt16, num_bandit), (x) -> alpha_value_estimator(convert(Float32, alpha), x, initial_value),
                                                                                (x) -> greedy_epsilon_selector(convert(Float32, epsilon), x), min_reward, max_reward)))
        end
        return rewards
end



