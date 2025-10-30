# YOUR CODE HERE
import sys

# Function to encode state tuple (v0, ..., vK-1) into an index
# State (v0, ..., vK-1) where vi is the capped value for parameter i+1
# Mapping: index = sum(vi * (P+1)^i for i=0 to K-1)
def encode_state(v, P):
    idx = 0
    p = 1
    for val in v:
        idx += val * p
        p *= (P + 1)
    return idx

# Function to decode index into state tuple (v0, ..., vK-1)
def decode_state(idx, K, P):
    v = [0] * K
    current_idx = idx
    for j in range(K):
        v[j] = current_idx % (P + 1)
        current_idx //= (P + 1)
    return v

def solve():
    # Read input N, K, P
    N, K, P = map(int, sys.stdin.readline().split())

    # Calculate the size of the DP table (P+1)^K
    state_size = (P + 1)**K

    # Initialize DP table: dp[state] = minimum cost to reach 'state'
    # State is represented by an index derived from (v0, ..., vK-1)
    # Initialize with infinity, except for the initial state (0, ..., 0) which has cost 0
    dp = [float('inf')] * state_size
    dp[0] = 0 # Initial state (0, 0, ..., 0) corresponds to index 0

    # Process each development plan
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        cost = line[0]
        # The increases A_{i,1}, ..., A_{i,K} are given for parameters 1 to K.
        # Store them in a 0-indexed list: increase[0] for param 1, ..., increase[K-1] for param K.
        increase = line[1:]

        # Iterate through all possible states (indices) in reverse order.
        # This ensures that when we update dp[next_idx], dp[idx] holds the cost
        # BEFORE considering the current plan applied FROM state idx.
        for idx in range(state_size - 1, -1, -1):
            # If the current state is unreachable, it cannot be a starting point
            if dp[idx] == float('inf'):
                continue

            # Decode the current state index into the tuple of capped parameter values (v0, ..., vK-1)
            current_v = decode_state(idx, K, P)

            # Calculate the next state and its index after applying the current plan
            next_idx_val = 0
            p = 1 # power of (P+1)

            for j in range(K):
                # The increase for parameter j+1 is increase[j]
                # The current capped value for parameter j+1 is current_v[j]
                new_val = min(current_v[j] + increase[j], P)

                # Calculate the next index on the fly using the new capped value
                next_idx_val += new_val * p
                p *= (P + 1)

            # Update the minimum cost to reach the next state (next_idx_val)
            dp[next_idx_val] = min(dp[next_idx_val], dp[idx] + cost)

    # The target state is where all parameters are at least P.
    # In our capped state representation, this is (P, P, ..., P).
    # Calculate the index for the target state (P, P, ..., P)
    # The index for (P, P, ..., P) using our mapping is sum(P * (P+1)^i for i=0 to K-1)
    # This sum is P * ((P+1)^K - 1) / P = (P+1)^K - 1 = state_size - 1
    target_idx = state_size - 1

    # The minimum cost to achieve the goal is stored at the target state index
    min_total_cost = dp[target_idx]

    # If the target state is unreachable, the value will still be infinity
    if min_total_cost == float('inf'):
        print(-1)
    else:
        print(min_total_cost)

# Execute the solve function
solve()