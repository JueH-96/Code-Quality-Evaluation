# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def count_arithmetic_subsequences(N, A):
    dp = [defaultdict(int) for _ in range(N)]
    count = [0] * (N + 1)
    
    for i in range(N):
        count[1] += 1
        for j in range(i):
            diff = A[i] - A[j]
            dp[i][diff] += dp[j][diff] + 1
            count[2] += dp[j][diff] + 1
    
    for k in range(3, N + 1):
        for i in range(N):
            for diff in dp[i]:
                count[k] += dp[i][diff]
                count[k] %= MOD
    
    return count[1:]

N = int(input())
A = list(map(int, input().split()))
result = count_arithmetic_subsequences(N, A)
print(' '.join(map(str, result)))