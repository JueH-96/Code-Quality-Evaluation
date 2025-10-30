# YOUR CODE HERE
MOD = 998244353

def count_strings(K, C):
    dp = [0] * (K + 1)
    dp[0] = 1
    
    for i in range(26):
        max_count = C[i]
        new_dp = [0] * (K + 1)
        for j in range(K + 1):
            for k in range(max_count + 1):
                if j + k <= K:
                    new_dp[j + k] = (new_dp[j + k] + dp[j]) % MOD
        dp = new_dp
    
    result = sum(dp[1:]) % MOD
    return result

import sys
input = sys.stdin.read
data = input().split()
K = int(data[0])
C = list(map(int, data[1:]))

print(count_strings(K, C))