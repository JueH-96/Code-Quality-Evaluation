def main():
    import sys
    sys.setrecursionlimit(10000)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # parse input
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    
    # build graph as an adjacency list, storing tuples (neighbor, weight)
    graph = {i: [] for i in range(1, N+1)}
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # initialize global result as large value
    INF = 1 << 65  # larger than maximum possible XOR (2^60)
    best = [INF]
    
    # Use DFS to find all simple path from 1 to N
    def dfs(current, xor_val, visited):
        # If reached N, update result
        if current == N:
            if xor_val < best[0]:
                best[0] = xor_val
            return
        
        # Explore neighbors
        for neighbor, w in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, xor_val ^ w, visited)
                visited.remove(neighbor)
    
    # Start from vertex 1
    visited = set()
    visited.add(1)
    dfs(1, 0, visited)
    
    # Output the result
    sys.stdout.write(str(best[0]))
    
if __name__ == '__main__':
    main()