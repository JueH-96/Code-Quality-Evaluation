def max_weight_matching(N, weights):
    from itertools import combinations

    # Create a list of all edges with their weights
    edges = []
    index = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            edges.append((weights[index], i, j))
            index += 1

    # Sort edges by weight in descending order
    edges.sort(reverse=True, key=lambda x: x[0])

    max_weight = 0
    chosen_edges = set()

    # Try to select edges while ensuring no endpoints are repeated
    for weight, u, v in edges:
        if u not in chosen_edges and v not in chosen_edges:
            chosen_edges.add(u)
            chosen_edges.add(v)
            max_weight += weight

    return max_weight

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
weights = []

for i in range(1, N):
    weights.extend(map(int, data[i].split()))

result = max_weight_matching(N, weights)
print(result)