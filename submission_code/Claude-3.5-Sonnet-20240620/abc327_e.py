# YOUR CODE HERE
import sys
import math

def calculate_rating(performances):
    n = len(performances)
    sum_numerator = sum((0.9 ** (n - i - 1)) * p for i, p in enumerate(performances))
    sum_denominator = sum(0.9 ** (n - i - 1) for i in range(n))
    return sum_numerator / sum_denominator - 1200 / math.sqrt(n)

def max_rating(N, P):
    max_r = float('-inf')
    for mask in range(1, 1 << N):
        selected = [P[i] for i in range(N) if mask & (1 << i)]
        r = calculate_rating(selected)
        max_r = max(max_r, r)
    return max_r

N = int(input())
P = list(map(int, input().split()))

result = max_rating(N, P)
print(f"{result:.15f}")