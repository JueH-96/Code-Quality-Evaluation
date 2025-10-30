from itertools import combinations
from math import sqrt

N = int(input())
points = [tuple(map(int, input().split())) for _ in range(N)]

def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

min_s = float('inf')
for C in range(N-1):
    for combo in combinations(range(1, N-1), C):
        dist = distance(points[0], points[1])
        for i in range(1, len(combo)):
            dist += distance(points[combo[i-1]+1], points[combo[i]+1])
        dist += distance(points[combo[-1]+1], points[N-1])
        penalty = 2 ** (C - 1) if C > 0 else 0
        min_s = min(min_s, dist + penalty)

print(min_s)