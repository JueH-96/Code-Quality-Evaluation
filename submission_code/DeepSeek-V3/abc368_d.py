import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    V = list(map(int, sys.stdin.readline().split()))
    
    # Find the farthest node from any V
    def bfs(start):
        visited = [-1] * (N+1)
        q = deque()
        q.append(start)
        visited[start] = 0
        farthest = start
        max_dist = 0
        while q:
            u = q.popleft()
            for v in edges[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    q.append(v)
                    if visited[v] > max_dist:
                        max_dist = visited[v]
                        farthest = v
        return farthest, max_dist
    
    # Find the diameter of the tree formed by the K nodes
    # First, find the farthest node from any V
    start = V[0]
    farthest, _ = bfs(start)
    # Then, find the farthest node from this farthest node
    farthest2, diameter = bfs(farthest)
    
    # The minimum number of nodes is the number of nodes in the path between farthest and farthest2
    # which is diameter + 1
    # But since we need to include all K nodes, we need to find the minimal subtree that contains all K nodes
    # The minimal subtree is the union of all paths between pairs of K nodes
    # The number of nodes in this subtree is the number of nodes in the union of all paths between pairs of K nodes
    # To find this, we can find the LCA of all pairs and then count the unique nodes in the paths
    
    # Precompute parent and depth for all nodes
    parent = [0] * (N+1)
    depth = [0] * (N+1)
    def dfs(u, p):
        parent[u] = p
        depth[u] = depth[p] + 1
        for v in edges[u]:
            if v != p:
                dfs(v, u)
    dfs(V[0], 0)
    
    # Function to find LCA
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        for _ in range(depth[u] - depth[v]):
            u = parent[u]
        if u == v:
            return u
        while parent[u] != parent[v]:
            u = parent[u]
            v = parent[v]
        return parent[u]
    
    # Find the LCA of all K nodes
    lca_all = V[0]
    for v in V[1:]:
        lca_all = lca(lca_all, v)
    
    # Now, for each V, find the path from V to lca_all and count unique nodes
    unique_nodes = set()
    for v in V:
        current = v
        while current != lca_all:
            unique_nodes.add(current)
            current = parent[current]
        unique_nodes.add(lca_all)
    
    print(len(unique_nodes))

if __name__ == "__main__":
    main()