# YOUR CODE HERE
MOD = 998244353

def power(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def solve(N, M, K, A):
    from collections import Counter
    from itertools import accumulate

    # Count the number of 1s in each bit position
    bit_count = [0] * 20
    for num in A:
        for i in range(20):
            if num & (1 << i):
                bit_count[i] += 1

    # Calculate the contribution of each bit position
    total_score = 0
    for i in range(20):
        if bit_count[i] == 0:
            continue

        # Calculate the number of subsequences of each length that include the i-th bit
        dp = [0] * (N + 1)
        dp[0] = 1
        for j in range(1, N + 1):
            dp[j] = (dp[j - 1] * 2 + (1 if bit_count[i] >= j else 0)) % MOD

        # Calculate the contribution of the i-th bit
        for length in range(M, N + 1, M):
            if bit_count[i] >= length:
                count = dp[length]
                score = power(1 << i, K, MOD)
                total_score = (total_score + count * score) % MOD

    return total_score

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
K = int(input[2])
A = list(map(int, input[3:]))

print(solve(N, M, K, A))