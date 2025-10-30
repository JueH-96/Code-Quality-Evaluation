# YOUR CODE HERE
import sys

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Compute prefix sums
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    # To handle the case where K is zero, we need to ensure that no subsequence sums to zero
    # But in the general case, we need to ensure that no subsequence sum equals K
    
    # We will use dynamic programming to count the number of valid divisions
    # dp[i] represents the number of ways to divide the first i elements such that no subsequence sum equals K
    
    # Initialize dp[0] = 1 (empty sequence)
    dp = [0] * (N + 1)
    dp[0] = 1
    
    # We will also use a dictionary to keep track of the last occurrence of each prefix sum
    last = {}
    last[0] = 0
    
    for i in range(1, N+1):
        # The current prefix sum is prefix[i]
        # We need to find all j < i such that prefix[i] - prefix[j] == K
        # So, prefix[j] = prefix[i] - K
        target = prefix[i] - K
        if target in last:
            # The last occurrence of target is last[target]
            # We need to subtract the number of ways that include a subsequence sum K
            # So, we subtract dp[last[target]]
            dp[i] = (dp[i-1] - dp[last[target]]) % MOD
        else:
            dp[i] = dp[i-1] % MOD
        # Update the last occurrence of the current prefix sum
        last[prefix[i]] = i
    
    # The answer is dp[N], but we need to ensure that it is non-negative
    print((dp[N] + MOD) % MOD)

if __name__ == "__main__":
    main()