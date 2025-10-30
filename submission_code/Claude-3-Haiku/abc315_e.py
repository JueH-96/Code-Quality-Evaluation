from collections import defaultdict

# Read input
N = int(input())
prerequisites = [[] for _ in range(N)]
for i in range(N):
    line = list(map(int, input().split()))
    C_i = line[0]
    for j in range(1, C_i + 1):
        prerequisites[i].append(line[j])

# Topological sort
visited = [False] * N
order = []

def dfs(node):
    visited[node - 1] = True
    for prereq in prerequisites[node - 1]:
        if not visited[prereq - 1]:
            dfs(prereq)
    order.append(node)

dfs(1)
order.reverse()

# Print the result
print(" ".join(map(str, order[1:])))