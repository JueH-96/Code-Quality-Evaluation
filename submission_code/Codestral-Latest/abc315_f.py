import math
import sys
from itertools import combinations

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def total_distance(path, points):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(points[path[i]][0], points[path[i]][1], points[path[i + 1]][0], points[path[i + 1]][1])
    return dist

def min_distance(points):
    n = len(points)
    min_dist = float('inf')

    for r in range(2, n):
        for combo in combinations(range(1, n - 1), r):
            path = [0] + list(combo) + [n - 1]
            dist = total_distance(path, points)
            penalty = 2 ** (len(combo) - 1) if len(combo) > 0 else 0
            total = dist + penalty
            if total < min_dist:
                min_dist = total

    return min_dist

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    points = []
    for i in range(N):
        x, y = int(data[2 * i + 1]), int(data[2 * i + 2])
        points.append((x, y))

    result = min_distance(points)
    print(result)

if __name__ == "__main__":
    main()