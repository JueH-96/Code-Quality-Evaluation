# YOUR CODE HERE
from math import comb
from bisect import bisect_left

def find_sequence(N, K):
    sequence = []
    remaining = [K] * N
    for i in range(N * K):
        index = bisect_left([comb(N * K - j, remaining[j]) for j in range(N)], (i + 1) // 2 + (i + 1) % 2)
        sequence.append(index + 1)
        remaining[index] -= 1
    return sequence

N, K = map(int, input().split())
result = find_sequence(N, K)
print(' '.join(map(str, result)))