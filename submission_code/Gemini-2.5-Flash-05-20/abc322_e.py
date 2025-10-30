import sys

# Define infinity for unreachable states
INF = float('inf')

# Helper function to convert a K-tuple state (v_0, v_1, ..., v_{K-1}) to a single integer index.
# This uses a base-(P_val+1) representation, where v_0 is the most significant digit.
def state_to_index(state_tuple, P_val, K_val):
    index = 0
    multiplier = 1
    # Iterate from the least significant parameter (v_{K-1}) to the most significant (v_0)
    for i in reversed(range(K_val)):
        index += state_tuple[i] * multiplier
        multiplier *= (P_val + 1)
    return index

# Helper function to convert a single integer index back to a K-tuple state.
def index_to_state(idx, P_val, K_val):
    state_list = [0] * K_val
    base = (P_val + 1)
    # Iterate from the least significant parameter (v_{K-1}) to the most significant (v_0)
    for i in reversed(range(K_val)):
        state_list[i] = idx % base
        idx //= base
    return tuple(state_list)

def solve():
    # Read N, K, P from standard input
    N, K, P = map(int, sys.stdin.readline().split())

    # Read all development plans
    plans = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        cost = line[0]
        # A_i,j values are 0-indexed in the code for consistency
        increases = line[1:] 
        plans.append((cost, increases))

    # Calculate the total number of possible states for the DP table.
    # Each parameter can take values from 0 to P, so P+1 distinct values.
    # Total states = (P+1)^K
    num_states = (P + 1) ** K

    # Initialize the DP table.
    # dp[index] stores the minimum cost to reach the state represented by 'index'.
    # All states are initially set to infinity, meaning they are unreachable.
    dp = [INF] * num_states

    # The starting state (all parameters are 0) has a cost of 0.
    initial_state_tuple = tuple(0 for _ in range(K))
    initial_idx = state_to_index(initial_state_tuple, P, K)
    dp[initial_idx] = 0

    # Iterate through each development plan
    for cost_i, increases_i in plans:
        # Create a new DP table for updates in this iteration.
        # This is critical for the "at most once" constraint. We derive new_dp from dp,
        # ensuring that any cost considered for plan 'i' is based on combinations of plans
        # processed prior to 'i'.
        next_dp = list(dp) # Make a shallow copy of the current dp values

        # Iterate through all possible previous states (represented by their flat index)
        for current_idx in range(num_states):
            # If the current state is unreachable, it cannot be a starting point for a new path.
            if dp[current_idx] == INF:
                continue

            # Convert the current flat index back to its K-tuple state
            current_state_tuple = index_to_state(current_idx, P, K)
            
            # Calculate the new state after applying the current plan's increases
            new_state_list = [0] * K
            for j in range(K):
                # The parameter value should be capped at P, as exceeding P provides no further benefit.
                new_state_list[j] = min(P, current_state_tuple[j] + increases_i[j])
            
            # Convert the new K-tuple state back to a flat index
            new_state_tuple = tuple(new_state_list)
            new_idx = state_to_index(new_state_tuple, P, K)

            # Update the minimum cost for the new state.
            # We take the minimum between the cost already recorded for new_idx (from other paths or previous plans)
            # and the cost of reaching current_idx plus the cost of the current plan.
            next_dp[new_idx] = min(next_dp[new_idx], dp[current_idx] + cost_i)
        
        # After processing all possible transitions for the current plan, update the main DP table.
        dp = next_dp

    # After iterating through all plans, find the minimum cost to achieve the goal.
    # The goal is to have all parameters at least P, which corresponds to the state (P,P,...,P)
    # in our capped DP table.
    final_target_state_tuple = tuple(P for _ in range(K))
    final_target_idx = state_to_index(final_target_state_tuple, P, K)

    result = dp[final_target_idx]

    # If the final target state is still INF, it means it's impossible to achieve the goal.
    if result == INF:
        print(-1)
    else:
        print(result)

# Call the main solve function to run the program
if __name__ == '__main__':
    solve()