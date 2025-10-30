import math

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        N = int(input())
        P = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handles cases with no input or malformed input.
        return

    # dp[j] will store the maximum weighted sum for choosing j contests.
    # The weighted sum for a subsequence Q_1, ..., Q_j is Sum_{l=1 to j} (0.9)^{j-l} * Q_l.
    # We use a 1D array for DP to save space, which requires iterating j backwards.
    dp = [float('-inf')] * (N + 1)
    dp[0] = 0.0

    # Iterate through each contest P_i
    for i in range(N):
        p_i = P[i]
        # The i-th contest (0-indexed) can be chosen as the j-th contest in a subsequence,
        # where j can be at most i+1.
        for j in range(i + 1, 0, -1):
            # For each j, we decide whether to include p_i or not.
            # If we don't include p_i, the max sum for j contests is the same as
            # the max sum for j contests from the first i-1 contests (dp[j] before this update).
            # If we include p_i as the j-th contest, we must have chosen j-1 contests
            # from the first i-1. The max sum for that was dp[j-1]. The new sum becomes
            # 0.9 * (sum for j-1 contests) + p_i.
            dp[j] = max(dp[j], 0.9 * dp[j-1] + p_i)

    max_rating = float('-inf')
    
    # The denominator for k chosen contests is Sum_{i=0 to k-1} (0.9)^i.
    # We can calculate this iteratively.
    current_den_sum = 0.0

    # Calculate the rating for each possible number of chosen contests, k.
    for k in range(1, N + 1):
        # Update the denominator sum: S_k = S_{k-1} * 0.9 + 1.0, with S_0 = 0.
        current_den_sum = current_den_sum * 0.9 + 1.0
        
        # The numerator is the max weighted sum for k contests, which we calculated with DP.
        numerator = dp[k]
        denominator = current_den_sum
        
        # The rating formula.
        rating = numerator / denominator - 1200 / math.sqrt(k)
        
        # Keep track of the maximum rating found.
        max_rating = max(max_rating, rating)
            
    print(f"{max_rating:.17f}")

solve()