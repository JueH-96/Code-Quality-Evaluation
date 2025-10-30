from collections import defaultdict

# Read input
N = int(input())
tree = defaultdict(list)

for _ in range(N-1):
    u, v = map(int, input().split())
    tree[u].append(v)
    tree[v].append(u)

# Compute the degree of each vertex
degrees = {}
for vertex in range(1, N+1):
    degrees[vertex] = len(tree[vertex])

# Count the number of vertices with each degree
degree_counts = defaultdict(int)
for degree in degrees.values():
    degree_counts[degree] += 1

# Compute the number of original stars
D_1 = degree_counts.get(1, 0)
M = (N + 2 - D_1) // 3

# Compute the levels of the original stars
levels = []

# For stars with level ≥ 3
for k in range(3, N+1):
    levels.extend([k] * degree_counts.get(k, 0))

# For stars with level 2
M_2 = M - sum(degree_counts.get(k, 0) for k in range(3, N+1))
levels.extend([2] * M_2)

# Output the multiset of levels in ascending order
levels.sort()
print(" ".join(map(str, levels)))