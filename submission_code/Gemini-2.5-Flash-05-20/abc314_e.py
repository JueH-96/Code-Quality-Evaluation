import sys

def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())

    wheels = []
    # Read data for each of the N wheels
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        C_i = line[0]        # Cost of playing this wheel
        P_i = line[1]        # Number of outcomes for this wheel
        S_values = line[2:]  # List of point outcomes for this wheel
        wheels.append({'C': C_i, 'P': P_i, 'S': S_values})

    # dp[k] will store the minimum expected cost to reach M points,
    # given that we currently have k points.
    # Initialize dp array. dp[M] (and beyond, effectively) is 0.0 because the goal is reached.
    # For k < M, dp[k] values will be calculated.
    dp = [0.0] * (M + 1)

    # Iterate k from M-1 down to 0.
    # This order ensures that when calculating dp[k], any dp[k'] where k' > k
    # are already correctly computed (or are 0 if k' >= M).
    for k in range(M - 1, -1, -1):
        min_expected_cost_for_k = float('inf') # Initialize with a very large number

        # Consider each available wheel as a choice from the current state k
        for wheel in wheels:
            C_i = wheel['C']
            P_i = wheel['P']
            S_values = wheel['S']

            sum_dp_values_for_nonzero_outcomes = 0.0
            count_zero_outcomes = 0

            # Calculate the sum of dp values for outcomes where points increase (s_val > 0)
            # and count of zero point outcomes (s_val == 0) for the current wheel.
            for s_val in S_values:
                if s_val == 0:
                    count_zero_outcomes += 1
                else:
                    # The new score should not exceed M, as points beyond M are
                    # effectively M for reaching the goal.
                    new_score = min(M, k + s_val)
                    sum_dp_values_for_nonzero_outcomes += dp[new_score]
            
            # The formula derived from solving for dp[k] with self-loops:
            # dp[k] = (C_i * P_i + sum_dp_values_for_nonzero_outcomes) / (P_i - count_zero_outcomes)

            denominator = P_i - count_zero_outcomes
            
            # The problem constraint `sum(S_i,j) > 0` for each wheel guarantees that
            # at least one S_i,j is > 0. Thus, `count_zero_outcomes` will always be
            # less than `P_i`, ensuring `denominator` is always greater than 0.
            # So, division by zero is not a concern.
            
            cost_if_choosing_this_wheel = (C_i * P_i + sum_dp_values_for_nonzero_outcomes) / denominator
            
            # Update the minimum expected cost found so far for state k
            min_expected_cost_for_k = min(min_expected_cost_for_k, cost_if_choosing_this_wheel)
        
        # Store the calculated minimum expected cost for state k
        dp[k] = min_expected_cost_for_k

    # The final answer is the minimum expected cost starting from 0 points.
    # Print the result formatted to 15 decimal places for precision.
    sys.stdout.write(f"{dp[0]:.15f}
")

# Call the solve function to execute the solution
if __name__ == '__main__':
    solve()