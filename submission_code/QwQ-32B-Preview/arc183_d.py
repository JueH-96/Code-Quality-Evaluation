import sys
from collections import deque

def read_ints(): return list(map(int, sys.stdin.read().split()))
def bfs(farthest, adj, dist):
    queue = deque([farthest])
    dist[farthest] = 0
    parent = [-1] * (N + 1)
    while queue:
        current = queue.popleft()
        for neighbor in adj[current]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[current] + 1
                parent[neighbor] = current
                queue.append(neighbor)
    max_dist = max(dist)
    farthest_leaf = dist.index(max_dist)
    return farthest_leaf, parent

def find_farthest_leaves(adj, leaves):
    # Start from the first leaf
    start = leaves[0]
    dist = [-1] * (N + 1)
    farthest_leaf1, parent1 = bfs(start, adj, dist)
    dist = [-1] * (N + 1)
    farthest_leaf2, parent2 = bfs(farthest_leaf1, adj, dist)
    return farthest_leaf1, farthest_leaf2, parent2

def get_path(parent, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = parent[current]
    path.append(start)
    return path[::-1]

def remove_vertices(adj, degrees, vertices):
    for v in vertices:
        for neighbor in adj[v]:
            degrees[neighbor] -= 1
            if degrees[neighbor] == 1:
                leaves.append(neighbor)
        adj[v] = []

def main():
    global N
    data = read_ints()
    idx = 0
    N = data[idx]
    idx += 1
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    for _ in range(N - 1):
        a = data[idx]
        b = data[idx + 1]
        idx += 2
        adj[a].append(b)
        adj[b].append(a)
        degrees[a] += 1
        degrees[b] += 1
    operations = []
    leaves = [v for v in range(1, N + 1) if degrees[v] == 1]
    while leaves:
        dist = [-1] * (N + 1)
        farthest_leaf1, farthest_leaf2, parent = find_farthest_leaves(adj, leaves)
        path = get_path(parent, farthest_leaf1, farthest_leaf2)
        operations.append((farthest_leaf1, farthest_leaf2))
        remove_vertices(adj, degrees, [farthest_leaf1, farthest_leaf2])
        leaves = [v for v in leaves if degrees[v] == 1]
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()