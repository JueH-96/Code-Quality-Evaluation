# YOUR CODE HERE
N, X = map(int, input().split())
T = list(map(int, input().split()))

MOD = 998244353
inv_N = pow(N, MOD - 2, MOD)

# dp[t] = probability that a song starts at time t
dp = [0] * (X + 1)
dp[0] = 1

# Calculate dp values
for t in range(1, X + 1):
    for i in range(N):
        if t >= T[i]:
            dp[t] = (dp[t] + dp[t - T[i]] * inv_N) % MOD

# Calculate probability that song 1 is playing at time X + 0.5
prob = 0
for start_time in range(max(0, X + 1 - T[0]), X + 1):
    prob = (prob + dp[start_time] * inv_N) % MOD

print(prob)