# YOUR CODE HERE
import sys
from itertools import combinations

def max_weight_matching(n, weights):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, weights[i][j - i - 1]))
    
    def is_valid(matching):
        vertices = set()
        for u, v, _ in matching:
            if u in vertices or v in vertices:
                return False
            vertices.add(u)
            vertices.add(v)
        return True
    
    max_weight = 0
    for k in range(1, n // 2 + 1):
        for matching in combinations(edges, k):
            if is_valid(matching):
                weight = sum(w for _, _, w in matching)
                max_weight = max(max_weight, weight)
    
    return max_weight

n = int(sys.stdin.readline())
weights = []
for i in range(n - 1):
    weights.append(list(map(int, sys.stdin.readline().split())))

print(max_weight_matching(n, weights))