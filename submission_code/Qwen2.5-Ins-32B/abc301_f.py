# YOUR CODE HERE
def solve(S):
    MOD = 998244353
    dp = [1, 0, 0, 0]
    for i in range(len(S)):
        new_dp = [0] * 4
        if S[i] == '?' or S[i].isupper():
            for j in range(4):
                new_dp[j] += dp[j] * (26 if S[i] == '?' else 1)
                new_dp[j] %= MOD
        if S[i] == '?' or S[i].islower():
            for j in range(3):
                new_dp[j + 1] += dp[j] * (26 if S[i] == '?' else 1)
                new_dp[j + 1] %= MOD
        dp = new_dp
    return sum(dp) % MOD

S = input().strip()
print(solve(S))