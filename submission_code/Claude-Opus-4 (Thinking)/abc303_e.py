# YOUR CODE HERE
n = int(input())
adj = [[] for _ in range(n + 1)]

for _ in range(n - 1):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

degree = [0] * (n + 1)
for i in range(1, n + 1):
    degree[i] = len(adj[i])

# Count vertices with degree 1
n_1 = sum(1 for i in range(1, n + 1) if degree[i] == 1)

# Calculate total number of stars
M = (n - n_1 + 2) // 3

# Vertices with degree >= 3 are centers of stars with that level
stars = []
for i in range(1, n + 1):
    if degree[i] >= 3:
        stars.append(degree[i])

# Remaining stars must be level-2
k_2 = M - len(stars)
stars.extend([2] * k_2)

stars.sort()
print(' '.join(map(str, stars)))