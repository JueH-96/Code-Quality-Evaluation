MOD = 998244353

def modinv(a):
    return pow(a, MOD - 2, MOD)

N, K = map(int, input().split())
P = list(map(int, input().split()))

def overlap(i, j):
    # i, j are 1-indexed
    if i <= N - K + 1 and j <= K:
        return i
    elif i <= N - K + 1 and j > K:
        return max(0, i - j + K)
    elif i > N - K + 1 and j <= K:
        return N - K + 1
    else:  # i > N - K + 1 and j > K
        return N - j + 1

orig_inv_count = 0
S1 = 0  # sum of overlaps for original inversions
S0 = 0  # sum of overlaps for original non-inversions

for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        is_inversion = 1 if P[i - 1] > P[j - 1] else 0
        ov = overlap(i, j)
        
        orig_inv_count += is_inversion
        if is_inversion:
            S1 += ov
        else:
            S0 += ov

# Expected inversions = (2 * orig_inv_count * (N-K+1) - S1 + S0) / (2 * (N-K+1))
numerator = (2 * orig_inv_count * (N - K + 1) - S1 + S0) % MOD
denominator = (2 * (N - K + 1)) % MOD

result = (numerator * modinv(denominator)) % MOD
print(result)