import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]  # Using 1-based indexing
    edges = set()

    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        if u > v:
            u, v = v, u
        edges.add((u, v))

    degree = [0] * (N + 1)
    for u in range(1, N+1):
        degree[u] = len(adj[u])

    d2_nodes = [u for u in range(1, N+1) if degree[u] == 2]

    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    depth = [0] * (N + 1)

    # BFS to set parent and depth for each node
    def bfs(root):
        q = deque([root])
        visited[root] = True
        parent[root] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v] and v != parent[u]:
                    visited[v] = True
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    q.append(v)

    # Build parent and depth information
    bfs(1)

    # Function to get path from u to v using parent pointers
    def get_path(u, v):
        path = []
        pu, pv = u, v
        while pu != pv:
            if depth[pu] < depth[pv]:
                pv = parent[pv]
            elif depth[pu] > depth[pv]:
                pu = parent[pu]
            else:
                pu, pv = parent[pu], parent[pv]
        # Now find the path from u to LCA and v to LCA
        path_u = []
        curr = u
        while curr != pu:
            path_u.append(curr)
            curr = parent[curr]
        path_v = []
        curr = v
        while curr != pu:
            path_v.append(curr)
            curr = parent[curr]
        path = path_u + [pu] + path_v[::-1]
        return path

    valid_pairs = 0
    M = len(d2_nodes)
    edge_pairs = 0

    # Check each pair of D2 nodes
    for i in range(M):
        u = d2_nodes[i]
        for j in range(i+1, M):
            v = d2_nodes[j]
            # Check if edge exists between u and v
            if ((u, v) in edges) or ((v, u) in edges):
                continue
            # Get path from u to v
            path = get_path(u, v)
            # Check all nodes in path except u and v have degree 3
            valid = True
            for node in path[1:-1]:
                if degree[node] != 3:
                    valid = False
                    break
            if valid:
                valid_pairs += 1

    print(valid_pairs)

if __name__ == '__main__':
    main()