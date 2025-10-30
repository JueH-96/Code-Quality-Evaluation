# YOUR CODE HERE
MOD = 998244353

def mod_inverse(a, m):
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

def expected_position(N, K):
    if N == 1:
        return 1
    prob = [0] * (N + 1)
    prob[1] = 1
    for _ in range(K):
        new_prob = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j:
                    new_prob[j] = (new_prob[j] + prob[i] * (1 / (N * (N - 1)))) % MOD
            new_prob[i] = (new_prob[i] + prob[i] * (1 - (N - 1) / (N * (N - 1)))) % MOD
        prob = new_prob
    expected = sum(i * prob[i] for i in range(1, N + 1)) % MOD
    return expected

import sys
input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
print(expected_position(N, K))