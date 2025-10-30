from collections import defaultdict

def find_star_levels(N, edges):
    # Build the adjacency list
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Find the star levels
    star_levels = []
    for node in range(1, N+1):
        if len(adj_list[node]) == 1:
            level = 1
            for neighbor in adj_list[node]:
                if len(adj_list[neighbor]) == 2:
                    level = 2
                    break
            star_levels.append(level)

    # Sort the star levels in ascending order
    star_levels.sort()
    return star_levels

# Read the input
N = int(input())
edges = []
for _ in range(N-1):
    u, v = map(int, input().split())
    edges.append((u, v))

# Solve the problem
star_levels = find_star_levels(N, edges)
print(' '.join(map(str, star_levels)))