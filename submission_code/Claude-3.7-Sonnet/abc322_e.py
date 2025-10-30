def min_cost_to_achieve_goal(N, K, P, costs, effects):
    # Initialize DP table
    dp = {tuple([0] * K): 0}  # Initially, all parameters are 0, with no cost
    
    # For each development plan
    for i in range(N):
        cost = costs[i]
        effect = effects[i]
        
        # Make a copy of the current DP table to prevent concurrent modification
        current_dp = list(dp.items())
        
        # Update DP table
        for state, cur_cost in current_dp:
            # Calculate new state after applying the development plan
            new_state = tuple(min(P, state[j] + effect[j]) for j in range(K))
            
            # Update cost to reach the new state
            new_cost = cur_cost + cost
            if new_state not in dp or new_cost < dp[new_state]:
                dp[new_state] = new_cost
    
    # Find the minimum cost to achieve the goal
    min_cost = float('inf')
    for state, cost in dp.items():
        if all(p >= P for p in state):
            min_cost = min(min_cost, cost)
    
    if min_cost == float('inf'):
        return -1
    else:
        return min_cost

# Read input
N, K, P = map(int, input().split())
costs = []
effects = []

for i in range(N):
    line = list(map(int, input().split()))
    costs.append(line[0])
    effects.append(line[1:])

# Solve the problem
result = min_cost_to_achieve_goal(N, K, P, costs, effects)

# Print the result
print(result)