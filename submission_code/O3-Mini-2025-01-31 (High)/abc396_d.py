def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
        
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the undirected graph.
    graph = {i: [] for i in range(1, n+1)}
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # We'll use DFS to explore all simple paths from 1 to n.
    best = [10**100]  # Using a list to allow modification within the inner function.
    
    def dfs(node, current_xor, visited):
        if node == n:
            if current_xor < best[0]:
                best[0] = current_xor
            return
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, current_xor ^ weight, visited)
                visited.remove(neighbor)
                
    visited = set()
    visited.add(1)
    dfs(1, 0, visited)
    
    sys.stdout.write(str(best[0]))

if __name__ == '__main__':
    main()