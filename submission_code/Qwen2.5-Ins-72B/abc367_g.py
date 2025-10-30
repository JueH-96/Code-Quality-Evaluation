# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def solve(N, M, K, A):
    # Initialize the frequency dictionary for XOR results
    xor_freq = defaultdict(int)
    xor_freq[0] = 1

    # Calculate the frequency of each XOR result
    for a in A:
        new_xor_freq = defaultdict(int)
        for xor_val, freq in xor_freq.items():
            new_xor_freq[xor_val] += freq
            new_xor_freq[xor_val ^ a] += freq
        xor_freq = new_xor_freq

    # Calculate the total score
    total_score = 0
    for xor_val, freq in xor_freq.items():
        if freq % 2 == 0:
            continue
        for length in range(1, N + 1, M):
            score = pow(xor_val, K, MOD)
            total_score += score * (freq // 2) * (freq // 2 - 1) // 2
            total_score %= MOD

    return total_score

# Read input
input = sys.stdin.read
N, M, K, *A = map(int, input().split())

# Solve and print the result
print(solve(N, M, K, A))