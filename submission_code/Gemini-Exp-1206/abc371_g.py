def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
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
    
    ans = [0] * n
    for cycle in cycles:
        cycle_len = len(cycle)
        min_a_cycle = [float('inf')] * cycle_len
        
        for start in range(cycle_len):
            curr_a_cycle = []
            for i in range(cycle_len):
                curr_a_cycle.append(a[cycle[(start + i) % cycle_len]])
            
            if curr_a_cycle < min_a_cycle:
                min_a_cycle = curr_a_cycle
        
        for i in range(cycle_len):
            ans[cycle[i]] = min_a_cycle[i]
    
    print(*ans)

solve()