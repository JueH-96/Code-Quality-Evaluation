# YOUR CODE HERE
MOD = 998244353

def count_arithmetic_subsequences(N, A):
    dp = [[0] * N for _ in range(N)]
    count = [0] * (N + 1)
    
    for i in range(N):
        count[1] += 1
        for j in range(i):
            diff = A[i] - A[j]
            if diff < 0:
                continue
            dp[i][j] = 1
            if diff * (i - j) <= 10**18:
                dp[i][j] += dp[j].get(diff * (i - j), 0)
            count[i - j + 1] += dp[i][j]
            count[i - j + 1] %= MOD
    
    return count[1:]

import sys
input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

result = count_arithmetic_subsequences(N, A)
print(" ".join(map(str, result)))