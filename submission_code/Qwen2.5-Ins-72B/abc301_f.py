# YOUR CODE HERE
import sys

MOD = 998244353

def count_ddos_free_strings(s):
    n = len(s)
    dp = [0] * 4
    dp[0] = 1
    
    for char in s:
        new_dp = [0] * 4
        for i in range(4):
            if char == '?' or char.isupper():
                new_dp[i] += dp[i] * 26
                new_dp[i] %= MOD
            if char == '?' or char.islower():
                new_dp[i] += dp[i] * 26
                new_dp[i] %= MOD
        
        if char == '?' or char.isupper():
            new_dp[1] += dp[0]
            new_dp[1] %= MOD
        if char == '?' or char.islower():
            new_dp[2] += dp[1]
            new_dp[2] %= MOD
        if char == '?' or char.isupper():
            new_dp[3] += dp[2]
            new_dp[3] %= MOD
        
        dp = new_dp
    
    return (52 ** s.count('?') - dp[3]) % MOD

s = input().strip()
print(count_ddos_free_strings(s))