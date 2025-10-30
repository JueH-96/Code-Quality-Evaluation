def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    # We need to find the probability that song 1 is being played at X + 0.5 seconds
    # This means that the total time played before must be exactly X seconds
    # and the song playing at X must be song 1.
    
    # dp[i] will be the probability that exactly i seconds have been played
    dp = [0] * (X + 1)
    dp[0] = 1  # Initially, 0 seconds have passed with probability 1
    
    # Update dp array for each second up to X
    for t in range(X):
        if dp[t] > 0:
            for duration in T:
                if t + duration <= X:
                    dp[t + duration] = (dp[t + duration] + dp[t] / N) % MOD
    
    # The probability that song 1 is being played at X + 0.5 seconds
    # is the probability that we have exactly X seconds played and the next song is song 1
    probability = (dp[X] / N) % MOD
    
    # To find the modular inverse of N under MOD
    def mod_inv(a, m):
        m0, x0, x1 = m, 0, 1
        if m == 1:
            return 0
        while a > 1:
            q = a // m
            m, a = a % m, m
            x0, x1 = x1 - q * x0, x0
        if x1 < 0:
            x1 += m0
        return x1
    
    # Calculate result as (probability * mod_inv(N, MOD)) % MOD
    result = (probability * mod_inv(N, MOD)) % MOD
    
    # Output the result
    print(result)