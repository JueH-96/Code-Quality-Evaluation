import sys

MOD = 998244353
N, M, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Precompute powers of 2
pow2 = [1] * (N + 1)
for i in range(1, N + 1):
    pow2[i] = pow2[i - 1] * 2 % MOD

# Precompute powers of 2 mod M
pow2_mod_M = [1] * (N + 1)
for i in range(1, N + 1):
    pow2_mod_M[i] = pow2_mod_M[i - 1] * 2 % M

# Precompute the XOR of A[i] and A[j] for all i < j
xor_A = [[0] * 21 for _ in range(N)]
for i in range(N):
    for b in range(20):
        if A[i] & (1 << b):
            xor_A[i][b] = (pow2[b] - pow2[b - 1] + MOD) % MOD
        else:
            xor_A[i][b] = pow2[b]

# Compute the XOR of all subsequences of A
xor_subseq = [0] * (1 << N)
for i in range(1, 1 << N):
    xor_subseq[i] = xor_subseq[i & (i - 1)]
    b = N - 1 - bin(i).count('0')
    xor_subseq[i] = (xor_subseq[i] + xor_A[b][0]) % MOD
    for j in range(20):
        if i & (1 << b):
            xor_subseq[i] = (xor_subseq[i] + pow2_mod_M[j] * xor_A[b][j]) % MOD

# Compute the sum of the scores of all subsequences of A
score_sum = [0] * (1 << N)
for i in range(1, 1 << N):
    score_sum[i] = score_sum[i - 1]
    if bin(i).count('1') % M == 0:
        score_sum[i] = (score_sum[i] + pow(xor_subseq[i], K)) % MOD

# Compute the sum of the scores of all non-empty subsequences of A
nonempty_score_sum = [0] * (1 << N)
for i in range(1, 1 << N):
    nonempty_score_sum[i] = (nonempty_score_sum[i - 1] + score_sum[i]) % MOD

# Compute the sum of the scores of all non-empty subsequences of A
total_score_sum = (nonempty_score_sum[(1 << N) - 1] - nonempty_score_sum[(1 << N) - 2]) % MOD

print(total_score_sum)