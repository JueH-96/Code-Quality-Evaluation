def solve():
    n = int(input())
    a = list(map(int, input().split()))
    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        adj[i + 1].append(a[i])

    visited = [False] * (n + 1)
    recursion_stack = [False] * (n + 1)
    cycle = []

    def dfs(u):
        nonlocal cycle
        visited[u] = True
        recursion_stack[u] = True
        
        for v in adj[u]:
            if not visited[v]:
                if dfs(v):
                    cycle.append(u)
                    return True
            elif recursion_stack[v]:
                cycle.append(u)
                cycle.append(v)
                return True
        
        recursion_stack[u] = False
        return False

    for i in range(1, n + 1):
        if not visited[i]:
            if dfs(i):
                break

    cycle.reverse()
    start_index = cycle.index(cycle[-1])
    final_cycle = cycle[start_index:]
    
    print(len(final_cycle))
    print(*final_cycle)

solve()