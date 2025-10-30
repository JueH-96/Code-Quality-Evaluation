def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    
    p = [x - 1 for x in p]
    
    visited = [False] * n
    cycles = []
    
    for i in range(n):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p[curr]
            cycles.append(cycle)
            
    result = [0] * n
    
    for cycle in cycles:
        cycle_len = len(cycle)
        for i in range(cycle_len):
            result[cycle[i]] = cycle[(i + k % cycle_len) % cycle_len]
            
    result = [x + 1 for x in result]
    print(*result)

solve()