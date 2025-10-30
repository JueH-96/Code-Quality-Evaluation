import sys
from collections import defaultdict

def main():
    MOD = 998244353
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Initialize DP table
    dp = [[defaultdict(int) for _ in range(N)] for _ in range(N+1)]
    
    # Base case: subsequences of length 1
    for i in range(N):
        dp[1][i][0] = 1  # Difference is irrelevant for k=1
    
    # Fill DP table for lengths from 2 to N
    for k in range(2, N+1):
        for i in range(N):
            for j in range(i):
                d = A[i] - A[j]
                dp[k][i][d] = (dp[k][i][d] + dp[k-1][j][d]) % MOD
    
    # Calculate and print the results for each k from 1 to N
    result = []
    for k in range(1, N+1):
        total_k = 0
        for i in range(N):
            for d in dp[k][i]:
                total_k = (total_k + dp[k][i][d]) % MOD
        result.append(str(total_k))
    
    print(' '.join(result))

if __name__ == '__main__':
    main()