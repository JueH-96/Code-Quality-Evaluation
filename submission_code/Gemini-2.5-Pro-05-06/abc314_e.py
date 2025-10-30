import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    wheels_data = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        C_i = line[0]
        P_i = line[1]
        S_ij_list = line[2:]
        wheels_data.append({'C': C_i, 'P': P_i, 'S_list': S_ij_list})

    # dp[s] stores the minimum expected cost to reach at least M points,
    # given current score s.
    # Size M+1 for scores 0, ..., M.
    # dp[M] = 0.0 as M points reached means no more cost.
    dp = [0.0] * (M + 1)

    # Iterate s from M-1 down to 0
    for s in range(M - 1, -1, -1):
        dp[s] = float('inf') # Initialize with a large value

        # Consider playing each wheel
        for wheel_info in wheels_data:
            C_i = wheel_info['C']
            P_i = wheel_info['P']
            S_list_i = wheel_info['S_list']

            sum_expected_future_costs_for_nonzero_scores = 0.0
            num_zero_score_outcomes = 0

            for score_outcome in S_list_i:
                if score_outcome == 0:
                    num_zero_score_outcomes += 1
                else:
                    next_state_score = s + score_outcome
                    # If next_state_score >= M, future cost is dp[M]=0.
                    # Otherwise, dp[next_state_score].
                    # This is captured by dp[min(M, next_state_score)].
                    sum_expected_future_costs_for_nonzero_scores += dp[min(M, next_state_score)]
            
            num_nonzero_score_outcomes = P_i - num_zero_score_outcomes
            
            # Based on problem constraints (sum of S_ij > 0 for any wheel),
            # num_nonzero_score_outcomes will be > 0.
            # So, no division by zero here.
            cost_if_play_wheel_i = (P_i * C_i + sum_expected_future_costs_for_nonzero_scores) / num_nonzero_score_outcomes
            
            dp[s] = min(dp[s], cost_if_play_wheel_i)

    # The result is dp[0]
    # Output with required precision. Python's default float to string conversion
    # might not be enough for all judges, specifying format is safer.
    sys.stdout.write(f"{dp[0]:.17f}
")

if __name__ == '__main__':
    solve()