MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m - 2, m)

def solve():
    N, X = map(int, input().split())
    T = list(map(int, input().split()))
    
    # dp[t] = probability that song 1 starts at time t
    max_time = X + max(T) + 1
    dp = [0] * (max_time + 1)
    
    # Base case: at time 0, song 1 is chosen with probability 1/N
    dp[0] = modinv(N)
    
    # For each time point
    for t in range(1, max_time + 1):
        # Song 1 can start at time t if any song ended at time t
        prob_sum = 0
        for j in range(N):
            if t >= T[j]:
                # Song j could have started at time t - T[j]
                # Probability of choosing song 1 after song j ends is 1/N
                prob_sum = (prob_sum + dp[t - T[j]]) % MOD
        dp[t] = (prob_sum * modinv(N)) % MOD
    
    # Find probability that song 1 is playing at time X + 0.5
    # Song 1 is playing at X + 0.5 if it started at time t where:
    # t <= X and t + T[0] > X + 0.5
    # Since t is integer, this means: t <= X and t + T[0] >= X + 1
    # So: max(0, X + 1 - T[0]) <= t <= X
    
    result = 0
    start = max(0, X + 1 - T[0])
    for t in range(start, X + 1):
        result = (result + dp[t]) % MOD
    
    print(result)

solve()