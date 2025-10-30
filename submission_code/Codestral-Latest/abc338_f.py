import itertools
import sys
input = sys.stdin.read
data = input().split()

index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1

edges = []
for _ in range(m):
    u = int(data[index])
    index += 1
    v = int(data[index])
    index += 1
    w = int(data[index])
    index += 1
    edges.append((u, v, w))

def find_min_weight():
    min_weight = float('inf')

    # Generate all permutations of vertices
    for perm in itertools.permutations(range(1, n + 1)):
        if perm[0] != 1:  # Ensure the walk starts from vertex 1
            continue
        weight = 0
        valid = True
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            edge_found = False
            for edge in edges:
                if edge[0] == u and edge[1] == v:
                    weight += edge[2]
                    edge_found = True
                    break
            if not edge_found:
                valid = False
                break
        if valid:
            min_weight = min(min_weight, weight)

    return min_weight if min_weight != float('inf') else "No"

result = find_min_weight()
print(result)