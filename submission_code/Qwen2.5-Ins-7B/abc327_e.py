# YOUR CODE HERE
from itertools import combinations

def calculate_rating(perfs, k):
    numerator = sum((0.9)**(k-i) * perfs[i] for i in range(k))
    denominator = sum((0.9)**(k-i) for i in range(k))
    return numerator / denominator - 1200 / (k**0.5)

def solve(perfs):
    max_rating = float('-inf')
    n = len(perfs)
    for k in range(1, n + 1):
        for combo in combinations(perfs, k):
            rating = calculate_rating(combo, k)
            if rating > max_rating:
                max_rating = rating
    return max_rating

n = int(input())
perfs = list(map(int, input().split()))
print(solve(perfs))