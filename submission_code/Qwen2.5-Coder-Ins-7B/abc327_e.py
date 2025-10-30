import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:]))

# Sort performances in descending order
P.sort(reverse=True)

max_rating = float('-inf')

# Iterate over all possible subsets of contests
for k in range(1, N + 1):
    numerator = 0
    denominator = 0
    for i in range(k):
        numerator += (0.9 ** i) * P[i]
        denominator += 0.9 ** i
    rating = (numerator / denominator) - (1200 / math.sqrt(k))
    if rating > max_rating:
        max_rating = rating

print(max_rating)