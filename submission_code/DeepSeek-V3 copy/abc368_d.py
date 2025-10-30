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
    
    # BFS to find the farthest node from any V
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
    # But we need to ensure that all K nodes are included in this path
    # So, we need to find the minimal subtree that contains all K nodes
    # This is equivalent to finding the minimal subtree that contains all K nodes, which is the union of all paths between pairs of K nodes
    # The minimal number of nodes is the number of nodes in the union of these paths
    
    # To find the minimal subtree, we can perform a BFS from all K nodes and mark the nodes that are in the union of all paths
    # However, a more efficient way is to find the LCA of all pairs of K nodes and then count the nodes in the union of paths from each K node to the LCA
    
    # First, find the LCA of all K nodes
    # We can use the Euler tour technique to find the LCA
    # Precompute the Euler tour and the depth of each node
    # Then, for any two nodes, the LCA is the node with the smallest depth in the Euler tour between the two nodes
    
    # Precompute the Euler tour
    euler = []
    depth = [0] * (N+1)
    first = [0] * (N+1)
    parent = [0] * (N+1)
    stack = [(start, False)]
    while stack:
        u, visited = stack.pop()
        if visited:
            euler.append(u)
        else:
            euler.append(u)
            stack.append((u, True))
            for v in edges[u]:
                if v != parent[u]:
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    stack.append((v, False))
    
    # Precompute the first occurrence of each node in the Euler tour
    for idx, node in enumerate(euler):
        if first[node] == 0:
            first[node] = idx
    
    # Function to find the LCA of two nodes
    def lca(u, v):
        if first[u] > first[v]:
            u, v = v, u
        # Find the node with the smallest depth in the Euler tour between first[u] and first[v]
        min_depth = float('inf')
        lca_node = u
        for i in range(first[u], first[v]+1):
            if depth[euler[i]] < min_depth:
                min_depth = depth[euler[i]]
                lca_node = euler[i]
        return lca_node
    
    # Find the LCA of all K nodes
    lca_all = V[0]
    for v in V[1:]:
        lca_all = lca(lca_all, v)
    
    # Now, count the number of nodes in the union of paths from each K node to the LCA
    # We can perform a BFS from the LCA and mark all nodes in the paths
    visited = set()
    q = deque()
    q.append(lca_all)
    visited.add(lca_all)
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v != parent[u] and v not in visited:
                visited.add(v)
                q.append(v)
    
    # Now, for each K node, perform a BFS from it to the LCA and mark all nodes in the path
    for v in V:
        current = v
        while current != lca_all:
            visited.add(current)
            current = parent[current]
    
    # The number of nodes in the minimal subtree is the size of the visited set
    print(len(visited))

if __name__ == "__main__":
    main()