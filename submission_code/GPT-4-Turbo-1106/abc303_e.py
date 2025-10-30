from collections import defaultdict, deque

def find_star_levels(N, edges):
    # Create an adjacency list for the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Find the degrees of all vertices
    degrees = [0] * (N + 1)
    for u in range(1, N + 1):
        degrees[u] = len(graph[u])

    # Find the leaves of the tree
    leaves = deque([i for i in range(1, N + 1) if degrees[i] == 1])

    # Count the levels of stars
    star_levels = []
    while leaves:
        leaf = leaves.popleft()
        # Decrease the degree of the neighbor
        neighbor = graph[leaf][0]
        degrees[neighbor] -= 1
        # If the neighbor becomes a leaf, add it to the leaves queue
        if degrees[neighbor] == 1:
            leaves.append(neighbor)
        # Count the level of the star
        star_level = degrees[leaf] + degrees[neighbor]
        if star_level > 1:
            star_levels.append(star_level - 1)
        # Remove the leaf from the graph
        graph[neighbor].remove(leaf)
        graph.pop(leaf)

    # Sort the levels of stars
    star_levels.sort()
    return star_levels

# Read input
N = int(input().strip())
edges = [tuple(map(int, input().strip().split())) for _ in range(N - 1)]

# Solve the problem
star_levels = find_star_levels(N, edges)

# Output the result
print(' '.join(map(str, star_levels)))