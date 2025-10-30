MOD = 998244353

def modinv(a, m):
    return pow(a, m - 2, m)

# Read input
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Convert to 0-indexed
P = [p - 1 for p in P]

# Compute the numerator (multiplied by 2)
numerator = 0

for a in range(N):
    for b in range(a + 1, N):
        # Compute C_{a,b} - number of positions where both a and b are shuffled
        if b - K + 1 > a:
            C_ab = 0
        else:
            C_ab = min(a, N - K) - max(0, b - K + 1) + 1
        
        if P[a] > P[b]:
            # This is an inversion
            numerator += 2 * (N - K + 1) - C_ab
        else:
            # This is not an inversion
            numerator += C_ab

denominator = 2 * (N - K + 1)

# Compute result modulo MOD
numerator = numerator % MOD
result = (numerator * modinv(denominator, MOD)) % MOD
print(result)