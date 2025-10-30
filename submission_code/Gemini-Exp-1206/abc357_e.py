def solve():
    n = int(input())
    a = list(map(int, input().split()))
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append(a[i] - 1)

    ans = 0
    for start_node in range(n):
        reachable = set()
        q = [start_node]
        reachable.add(start_node)
        
        while q:
            curr = q.pop(0)
            for neighbor in adj[curr]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    q.append(neighbor)
        ans += len(reachable)

    print(ans)

solve()