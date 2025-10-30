MOD = 998244353

def modinv(a):
    return pow(a, MOD-2, MOD)

N, X = map(int, input().split())
T = list(map(int, input().split()))

# dp[t] = probability distribution at time t
# dp[t][i] = probability that song i just started at time t
dp = [{} for _ in range(X + max(T) + 1)]

# At time 0, each song starts with probability 1/N
for i in range(N):
    dp[0][i] = modinv(N)

# For each time step
for t in range(X + max(T)):
    for song, prob in dp[t].items():
        # This song plays for T[song] seconds
        next_time = t + T[song]
        if next_time < len(dp):
            # When this song ends, start a new random song
            for next_song in range(N):
                if next_song not in dp[next_time]:
                    dp[next_time][next_song] = 0
                dp[next_time][next_song] = (dp[next_time][next_song] + prob * modinv(N)) % MOD

# Now calculate probability that song 0 is playing at time X + 0.5
result = 0

# Check all possible start times for song 0
for start_time in range(X + 1):
    if 0 in dp[start_time]:
        # Song 0 starts at start_time and plays for T[0] seconds
        end_time = start_time + T[0]
        # Check if X + 0.5 is within [start_time, end_time)
        if start_time <= X + 0.5 < end_time:
            result = (result + dp[start_time][0]) % MOD

print(result)