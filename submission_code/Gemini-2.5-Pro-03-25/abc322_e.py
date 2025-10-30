# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read N, K, P from the first line of input
    # N: number of development plans
    # K: number of parameters
    # P: target minimum value for each parameter
    N, K, P = map(int, sys.stdin.readline().split())
    
    plans = []
    # Read N lines of plan data
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        # Store each plan as a tuple: (cost, tuple_of_parameter_increases)
        # The first element is the cost C_i.
        # The rest are parameter increases A_{i,1}, ..., A_{i,K}.
        # Storing increases as a tuple A = (A_{i,1}, ..., A_{i,K}) for easy access.
        plans.append((line[0], tuple(line[1:])))

    # Use float('inf') to represent unreachable states or infinite cost.
    # This value is guaranteed to be larger than any possible finite cost derived from the input constraints.
    INF = float('inf')

    # Initialize the dynamic programming table using a dictionary.
    # The keys of the dictionary `dp` are state tuples (v1, ..., vK), representing the current values of the K parameters.
    # The values are the minimum cost found so far to reach that state.
    # Parameter values v_j are capped at P. If a parameter value reaches P or exceeds it, 
    # we treat it as P in the state representation. This keeps the state space manageable.
    # So, for any state tuple (v1, ..., vK) stored, 0 <= v_j <= P for all j.
    dp = {}
    
    # The initial state is (0, 0, ..., 0), where all K parameters start at zero.
    # The cost to reach this initial state is 0.
    initial_state = tuple([0] * K)
    dp[initial_state] = 0

    # Iterate through each of the N development plans
    for i in range(N):
        C, A = plans[i] # C is the cost of the i-th plan, A is the tuple of parameter increases (A_1, ..., A_K)
        
        # Create a copy of the current dp table. `dp_next` will store the minimum costs
        # after considering the i-th plan. It's initialized with `dp` values, which represents
        # the costs achievable without using the i-th plan. This way, we correctly take the minimum
        # over both options: using plan i or not using plan i to reach a state.
        dp_next = dp.copy() 

        # Iterate through all states currently reachable (i.e., states present as keys in `dp`)
        # `state_tuple` is the state reached using plans before i, `current_cost` is its minimum cost.
        for state_tuple, current_cost in dp.items():
            
            # Calculate the resulting state if plan i is applied starting from `state_tuple`
            next_state_list = list(state_tuple) # Convert tuple to list for modification
            for j in range(K):
                # For each parameter j, calculate its new value: current value + increase from plan i.
                current_val = state_tuple[j]
                increase = A[j] # A[j] corresponds to A_{i, j+1} from problem (1-based vs 0-based index)
                
                # Cap the new value at P. Any value >= P is treated as P.
                next_state_list[j] = min(current_val + increase, P) 
            
            # Convert the list back to a tuple to use it as a dictionary key. This is the new state reached.
            next_state_tuple = tuple(next_state_list)
            
            # Calculate the total cost to reach `next_state_tuple` via `state_tuple` using plan i.
            new_cost = current_cost + C 

            # Compare this `new_cost` with the currently known minimum cost to reach `next_state_tuple`.
            # `dp_next.get(next_state_tuple, INF)` retrieves this minimum cost. If `next_state_tuple`
            # hasn't been reached yet or updated in this iteration, it defaults to INF.
            existing_cost_for_next_state = dp_next.get(next_state_tuple, INF)
            
            # If the path using plan i results in a lower total cost (`new_cost`) to reach `next_state_tuple`,
            # update the cost in `dp_next`.
            if new_cost < existing_cost_for_next_state:
                dp_next[next_state_tuple] = new_cost
        
        # After iterating through all states in `dp` and calculating potential updates via plan i,
        # `dp_next` contains the minimum costs considering plans up to i.
        # Replace the old `dp` table with `dp_next` for the next iteration (plan i+1).
        dp = dp_next

    # The goal state is achieved when all K parameters have values of at least P.
    # Because we cap values at P, this corresponds exactly to the state (P, P, ..., P).
    target_state = tuple([P] * K)
    
    # Retrieve the minimum cost to reach the target state from the final dp table.
    # If `target_state` is not a key in `dp`, it means it was never reached, so `get` returns `INF`.
    min_cost = dp.get(target_state, INF)

    # Check if the target state was reachable.
    if min_cost == INF:
        # If `min_cost` is still `INF`, the goal cannot be achieved with any combination of plans. Print -1.
        print("-1")
    else:
        # Otherwise, the goal is achievable. Print the minimum cost found.
        # The cost is guaranteed to be an integer based on input constraints and operations.
        # We cast to `int` explicitly for clean output format. Python's integers handle arbitrary sizes.
        print(int(min_cost)) 

# Standard boilerplate for competitive programming main execution block.
# Ensures `solve()` is called only when the script is executed directly.
if __name__ == '__main__':
    solve()