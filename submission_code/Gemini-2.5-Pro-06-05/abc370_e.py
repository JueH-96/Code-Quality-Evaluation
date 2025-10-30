import sys

def main():
    """
    Reads input, solves the problem, and prints the answer.
    The problem is solved using dynamic programming with a hash map for optimization.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        line = readline()
        if not line:
            return
        
        N, K = map(int, line.split())
        A = list(map(int, readline().split()))
    except (IOError, ValueError):
        # This handles cases like empty input or malformed lines,
        # which are not expected in typical competitive programming scenarios.
        return

    MOD = 998244353

    # prefix_sums[i] will store the sum of A[0]...A[i-1]
    # We use 0-based indexing for A, so A_1..A_N corresponds to A[0]..A[N-1]
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + A[i]

    # dp[i]: number of valid divisions of the prefix A[0...i-1]
    dp = [0] * (N + 1)
    
    # Base case: dp[0] = 1. This represents a valid "empty" partition of the empty prefix.
    # It serves as the starting point for our DP.
    dp[0] = 1
    
    # total_dp_sum is the running sum of dp[j] for all j processed so far.
    total_dp_sum = 1 
    
    # map_sum_to_dp: maps a prefix sum value to the sum of dp values
    # that correspond to prefixes with that sum.
    # map_sum_to_dp[val] = sum(dp[j]) for all j where prefix_sums[j] == val
    map_sum_to_dp = {0: 1} # For prefix_sums[0]=0, we have dp[0]=1

    for i in range(1, N + 1):
        # We are calculating dp[i] based on previous dp values.
        
        # Identify the prefix sum value that would lead to a bad last segment.
        # A segment A[j...i-1] is "bad" if its sum is K.
        # sum = prefix_sums[i] - prefix_sums[j] = K  =>  prefix_sums[j] = prefix_sums[i] - K
        bad_prefix_sum_val = prefix_sums[i] - K
        
        # Get the sum of dp values for all such "bad" previous states j.
        bad_dp_sum = map_sum_to_dp.get(bad_prefix_sum_val, 0)
        
        # dp[i] is the total ways to partition up to i-1, minus the ways that end
        # with a bad segment. `total_dp_sum` currently holds sum of dp[j] for j < i.
        dp[i] = (total_dp_sum - bad_dp_sum + MOD) % MOD
        
        # Update the running total of dp values to include the newly calculated dp[i].
        total_dp_sum = (total_dp_sum + dp[i]) % MOD
        
        # Update the map with the new dp[i] value and its corresponding prefix sum.
        current_prefix_sum = prefix_sums[i]
        map_sum_to_dp[current_prefix_sum] = (map_sum_to_dp.get(current_prefix_sum, 0) + dp[i]) % MOD

    # The final answer is the number of valid divisions for the entire sequence A, which is dp[N].
    print(dp[N])

if __name__ == "__main__":
    main()