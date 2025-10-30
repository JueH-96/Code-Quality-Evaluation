def maximum_strength(n, k, arr):
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + arr[i - 1]
    
    dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
    
    for j in range(1, k + 1):
        sign = (-1) ** (j + 1)
        min_prefix = prefix[0]  # Initialize with prefix[0] which is 0
        max_val = -float('inf')
        for i in range(1, n + 1):
            # Calculate the contribution for the current subarray ending at i
            if j == 1:
                contribution = sign * (prefix[i] - min_prefix)
            else:
                # For j > 1, the max_val is the maximum of (dp[s][j-1] - sign * prefix[s]) for s < i
                contribution = sign * (prefix[i] - min_prefix)
                if dp[i - 1][j - 1] - sign * prefix[i - 1] > max_val:
                    max_val = dp[i - 1][j - 1] - sign * prefix[i - 1]
                    min_prefix = prefix[i - 1]
            # Update dp[i][j]
            if j == 1:
                dp[i][j] = max(dp[i - 1][j], contribution)
            else:
                current_candidate = max_val + sign * prefix[i]
                dp[i][j] = max(dp[i - 1][j], current_candidate)
    
    return dp[n][k]

# Read input
import sys
def main():
    n, k = map(int, sys.stdin.readline().split())
    arr = list(map(int, sys.stdin.readline().split()))
    print(maximum_strength(n, k, arr))

if __name__ == "__main__":
    main()