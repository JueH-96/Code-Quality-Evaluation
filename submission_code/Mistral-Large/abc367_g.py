import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

MOD = 998244353

# Precompute powers of K modulo MOD
powK = [1] * (N + 1)
for i in range(1, N + 1):
    powK[i] = powK[i - 1] * K % MOD

# Precompute XORs for all subsequences
xor_values = [0] * (N + 1)
for i in range(1, N + 1):
    xor_values[i] = xor_values[i - 1] ^ A[i - 1]

# Calculate the sum of scores
total_score = 0
for i in range(1, 1 << N):
    subset_size = bin(i).count('1')
    if subset_size % M == 0:
        xor_result = 0
        for j in range(N):
            if i & (1 << j):
                xor_result ^= A[j]
        total_score += pow(xor_result, K, MOD)
        total_score %= MOD

print(total_score)