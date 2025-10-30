def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    
    # Build adjacency list
    edges_data = data[2 : 2*(N-1) + 2]
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        a = int(edges_data[idx])
        b = int(edges_data[idx+1])
        idx += 2
        adj[a].append(b)
        adj[b].append(a)
    
    # Read required vertices
    vs_data = data[2*(N-1) + 2 : 2*(N-1) + 2 + K]
    required_nodes = [False]*(N+1)
    for v in vs_data:
        required_nodes[int(v)] = True
    
    # keep[u] will be True if node u is part of the minimal subtree
    keep = [False]*(N+1)
    
    def dfs(u, parent):
        needed = required_nodes[u]
        for nxt in adj[u]:
            if nxt == parent:
                continue
            if dfs(nxt, u):
                needed = True
        keep[u] = needed
        return needed
    
    # Root the DFS at one of the required vertices (e.g. vs_data[0])
    root = int(vs_data[0])
    dfs(root, -1)
    
    # Count how many nodes are kept
    answer = sum(keep)
    print(answer)

# Remember to call main() at the end
if __name__ == "__main__":
    main()