import sys
from math import factorial

MOD = 998244353

def solve():
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))

    # Compute the number of sequences of length N consisting of 1..K
    total_sequences = pow(K, N, MOD)

    # Compute the number of sequences that have the given subsequence (X1, ..., XM)
    num_bad_sequences = 0
    for i in range(1, N-M+2):
        num_bad_sequences += factorial(N-M) * pow(K-1, N-M, MOD)
        num_bad_sequences %= MOD

    # Compute the number of valid sequences
    return (total_sequences - num_bad_sequences) % MOD

print(solve())