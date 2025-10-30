def solve():
    n = int(input())
    a = list(map(int, input().split()))
    w = list(map(int, input().split()))

    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[a[i] - 1].append(i)

    visited = [False] * n
    ans = 0

    for start_node in range(n):
        if not visited[start_node]:
            cycle = []
            curr = start_node
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = a[curr] - 1

            cycle_nodes = []
            for i in range(len(cycle)):
                if cycle[i] == curr:
                    cycle_nodes = cycle[i:]
                    break
            
            if len(cycle_nodes) > 0:
                min_weight = float('inf')
                total_weight = 0
                for node in cycle_nodes:
                    min_weight = min(min_weight, w[node])
                    total_weight += w[node]
                
                ans += total_weight - min_weight
    
    print(ans)

solve()