import sys
from math import sqrt

def read_input():
    N = int(sys.stdin.readline().strip())
    P = list(map(int, sys.stdin.readline().strip().split()))
    return N, P

def calculate_rating(k, Q):
    numerator = sum(Q[i] * (0.9 ** (k - i)) for i in range(k))
    denominator = sum(0.9 ** (k - i) for i in range(k))
    return numerator / denominator - 1200 / sqrt(k)

def solve():
    N, P = read_input()
    max_rating = -float('inf')
    for i in range(N):
        for j in range(i + 1, N + 1):
            max_rating = max(max_rating, calculate_rating(j - i, P[i:j]))
    print(max_rating)

solve()