# YOUR CODE HERE
MOD = 998244353

def count_arithmetic_subsequences(N, A):
    dp = [{} for _ in range(N)]
    result = [0] * N
    
    for i in range(N):
        result[0] += 1
        for j in range(i):
            diff = A[i] - A[j]
            if diff in dp[j]:
                dp[i][diff] = (dp[i].get(diff, 0) + dp[j][diff] + 1) % MOD
            else:
                dp[i][diff] = (dp[i].get(diff, 0) + 1) % MOD
            result[dp[j][diff]] = (result[dp[j][diff]] + dp[j][diff]) % MOD
    
    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

result = count_arithmetic_subsequences(N, A)
print(" ".join(map(str, result)))