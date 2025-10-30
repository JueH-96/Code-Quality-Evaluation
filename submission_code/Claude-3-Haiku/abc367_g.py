import sys

def xor(a, b):
    return a ^ b

def score(seq):
    if len(seq) % M == 0:
        return pow(sum(seq), K, MOD)
    else:
        return 0

N, M, K = map(int, input().split())
A = list(map(int, input().split()))
MOD = 998244353

total = 0
for mask in range(1, 2 ** N):
    curr_seq = [A[i] for i in range(N) if (mask >> i) & 1]
    total = (total + score(curr_seq)) % MOD

print(total)