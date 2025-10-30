def mod_inverse(a, p):
    return pow(a, p - 2, p)

def calculate_probability(N, A):
    MOD = 998244353
    total_outcomes = 1
    for a in A:
        total_outcomes = total_outcomes * a % MOD

    dp = [0] * (10 + 1)
    dp[0] = 1  # There's one way to get sum 0: choose no dice

    for a in A:
        for j in range(10, -1, -1):
            for k in range(1, a + 1):
                if j >= k:
                    dp[j] = (dp[j] + dp[j - k]) % MOD

    successful_outcomes = dp[10]
    probability_numerator = successful_outcomes
    probability_denominator = total_outcomes

    z = (probability_numerator * mod_inverse(probability_denominator, MOD)) % MOD
    return z

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N + 1]))

result = calculate_probability(N, A)
print(result)