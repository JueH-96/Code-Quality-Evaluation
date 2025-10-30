# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Calculate prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
    
    # dp[i] will store the number of valid partitions up to the i-th element
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to partition an empty array
    
    # Map to store the count of prefix sums
    prefix_count = {}
    prefix_count[0] = 1  # There's one way to have a prefix sum of 0 (the empty prefix)
    
    for i in range(1, N + 1):
        # Calculate the number of valid partitions up to i
        dp[i] = dp[i - 1]
        
        # Check if there's a prefix sum that makes a subsequence sum to K
        target_prefix_sum = prefix_sum[i] - K
        if target_prefix_sum in prefix_count:
            dp[i] = (dp[i] - prefix_count[target_prefix_sum]) % MOD
        
        # Update the prefix_count with the current prefix_sum[i]
        if prefix_sum[i] in prefix_count:
            prefix_count[prefix_sum[i]] = (prefix_count[prefix_sum[i]] + dp[i]) % MOD
        else:
            prefix_count[prefix_sum[i]] = dp[i]
    
    # The answer is the number of valid partitions up to the N-th element
    print(dp[N])