def modinv(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

N, X = map(int, input().split())
T = list(map(int, input().split()))
MOD = 998244353

# For each song, find if it can be playing at time X+0.5
# by checking if X+0.5 falls within any of its possible play intervals
target_time = X + 0.5
numerator = 0
denominator = N

# For song 1
for start_time in range(int(target_time + 1)):
    # If we start song 1 at start_time
    end_time = start_time + T[0]
    if start_time <= target_time < end_time:
        # Calculate probability of reaching this start time
        # Probability is (1/N)^(number of songs needed to reach this time)
        songs_needed = start_time // min(T)
        if songs_needed == 0:
            numerator += pow(N, N-1, MOD)
        else:
            numerator += pow(N, N-songs_needed-1, MOD)

# Final probability is numerator/N^(N)
result = (numerator * modinv(pow(N, N-1, MOD), MOD)) % MOD
print(result)