import sys

def main():
    """
    Reads input, solves the problem using dynamic programming, and prints the output.
    """
    # Fast I/O
    input = sys.stdin.readline

    # Read problem parameters
    try:
        N, K, P = map(int, input().split())
    except ValueError:
        # Handles empty input during local testing
        return

    # Read development plans
    plans = []
    for _ in range(N):
        line = list(map(int, input().split()))
        cost, params = line[0], line[1:]
        plans.append((cost, params))

    # --- Dynamic Programming Setup ---
    # Base for state-to-index conversion
    B = P + 1
    # Total number of states is (P+1)^K
    dp_size = B ** K

    # Initialize DP table. dp[i] = min cost to reach state i.
    # Use a large number for infinity. Max cost is ~100 * 10^9.
    INF = float('inf')
    dp = [INF] * dp_size
    # Base case: cost to reach state (0, 0, ..., 0) is 0.
    dp[0] = 0

    # Pre-calculate powers of B for efficient state-to-index encoding.
    # index = s_0*B^(K-1) + s_1*B^(K-2) + ... + s_{K-1}
    powers_of_B = [1] * K
    for i in range(K - 2, -1, -1):
        powers_of_B[i] = powers_of_B[i + 1] * B

    # --- DP Transitions ---
    # Process each development plan
    for cost, params_increase in plans:
        # Iterate through states in reverse order for in-place DP updates.
        for idx in range(dp_size - 1, -1, -1):
            # If the current state is unreachable, skip.
            if dp[idx] == INF:
                continue

            # Decode the 1D index back to the K-dimensional state
            current_state = [0] * K
            temp_idx = idx
            for k in range(K - 1, -1, -1):
                current_state[k] = temp_idx % B
                temp_idx //= B

            # Calculate the new state after applying the current plan
            new_state = [0] * K
            for k in range(K):
                new_state[k] = min(P, current_state[k] + params_increase[k])

            # Encode the new state to its corresponding 1D index
            new_idx = 0
            for k in range(K):
                new_idx += new_state[k] * powers_of_B[k]

            # Update the minimum cost to reach the new state
            dp[new_idx] = min(dp[new_idx], dp[idx] + cost)

    # --- Final Result ---
    # The target state is (P, P, ..., P), which corresponds to the highest index.
    min_cost = dp[dp_size - 1]

    # If the target state is still INF, it's unreachable.
    if min_cost == INF:
        print(-1)
    else:
        print(int(min_cost))

if __name__ == "__main__":
    main()