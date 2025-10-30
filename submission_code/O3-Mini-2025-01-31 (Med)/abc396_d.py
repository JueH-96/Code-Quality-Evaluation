def main():
    import sys
    sys.setrecursionlimit(10**6)
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the undirected graph: dictionary from vertex to list of (neighbor, weight)
    graph = {i: [] for i in range(1, n + 1)}
    for _ in range(m):
        u = int(next(it)); v = int(next(it)); w = int(next(it))
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    best = None  # To store the best (minimum) XOR value found
    
    # DFS to explore all simple paths from 1 to n
    def dfs(v, xor_val, visited):
        nonlocal best
        if v == n:
            if best is None or xor_val < best:
                best = xor_val
            return
        for neighbor, w in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, xor_val ^ w, visited)
                visited.remove(neighbor)
    
    visited = {1}
    dfs(1, 0, visited)
    print(best)

if __name__ == '__main__':
    main()