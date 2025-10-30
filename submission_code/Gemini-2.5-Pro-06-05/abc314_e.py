# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Roulette problem using dynamic programming.
    """
    try:
        # Read problem parameters N (number of wheels) and M (target score)
        line = sys.stdin.readline()
        if not line:
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        # Handles potential empty input or format errors
        return

    # Pre-process and store data for each wheel to optimize DP calculation
    wheels = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        C, P = line[0], line[1]
        S = line[2:]

        # Count outcomes with 0 score and collect non-zero scores
        zero_score_count = S.count(0)
        non_zero_scores = [s for s in S if s > 0]
        
        # The problem guarantees sum(S_ij) > 0, so P - zero_score_count > 0.
        # This denominator is guaranteed to be non-zero.
        denominator = float(P - zero_score_count)
        
        # Pre-calculate the constant part of the numerator in the cost formula
        numerator_constant = float(P * C)

        wheels.append({
            'num_const': numerator_constant,
            'denom': denominator,
            'scores': non_zero_scores
        })

    # DP state: dp[k] = minimum expected cost to earn at least k more points.
    dp = [0.0] * (M + 1)
    # Base case: dp[0] = 0, as no more points are needed.

    # Iterate from k=1 to M to fill the DP table
    for k in range(1, M + 1):
        min_expected_cost_for_k = float('inf')

        # For the current state (k points needed), decide which wheel to play.
        # The optimal choice is the one that minimizes the expected cost.
        for wheel in wheels:
            # Calculate the sum of expected future costs from non-zero outcomes
            sum_future_dp = 0
            for score in wheel['scores']:
                # If k - score <= 0, we have reached the target, future cost is 0.
                # Otherwise, it's dp[k - score].
                # max(0, k - score) handles both cases.
                index = max(0, k - score)
                sum_future_dp += dp[index]
            
            # Calculate the total expected cost for choosing this wheel at state k.
            # This formula is derived by resolving the self-loop in the state transition
            # graph caused by 0-score outcomes.
            current_wheel_cost = (wheel['num_const'] + sum_future_dp) / wheel['denom']

            # Update the minimum expected cost for state k
            min_expected_cost_for_k = min(min_expected_cost_for_k, current_wheel_cost)

        # Store the optimal value in the DP table
        dp[k] = min_expected_cost_for_k

    # The final answer is the minimum expected cost to get at least M points.
    # Print the result with high precision to meet the problem's requirements.
    print(f"{dp[M]:.15f}")

if __name__ == "__main__":
    solve()