def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Prefix sums and dynamic programming
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]
    
    # dp[i] will store the number of valid partitions of the subarray A[0:i]
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: empty array has one valid partition (itself, doing nothing)
    
    # To quickly check if a sum has been seen before
    from collections import defaultdict
    sum_count = defaultdict(int)
    sum_count[0] = 1  # We can always start from an empty subarray
    
    for i in range(1, N + 1):
        # We are considering the array A[0:i]
        current_prefix = prefix_sum[i]
        
        # We want to find all j < i such that prefix_sum[j] == current_prefix - K
        # dp[i] should be incremented by dp[j] for all such j
        needed_sum = current_prefix - K
        if needed_sum in sum_count:
            dp[i] = sum_count[needed_sum]
        
        # Update the sum_count for future use
        sum_count[current_prefix] = (sum_count[current_prefix] + dp[i]) % MOD
    
    # The answer is dp[N], which considers the entire array A[0:N]
    print(dp[N])

if __name__ == "__main__":
    main()