import sys
import math

def calculate_rating(k, performances):
    numerator = sum((0.9 ** (k - i - 1)) * performances[i] for i in range(k))
    denominator = sum((0.9 ** (k - i - 1)) for i in range(k))
    penalty = 1200 / math.sqrt(k)
    return (numerator / denominator) - penalty

def max_rating(performances):
    n = len(performances)
    max_rate = -float('inf')
    for k in range(1, n + 1):
        for i in range(n - k + 1):
            current_rate = calculate_rating(k, performances[i:i + k])
            max_rate = max(max_rate, current_rate)
    return max_rate

input = sys.stdin.read
data = input().split()
N = int(data[0])
P = list(map(int, data[1:]))

print(f"{max_rating(P):.12f}")