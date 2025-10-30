def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append(a[i] - 1)
    
    count = 0
    for start_node in range(n):
        visited = set()
        q = [start_node]
        
        while q:
            curr = q.pop(0)
            if curr not in visited:
                visited.add(curr)
                count += 1
                for neighbor in adj[curr]:
                    q.append(neighbor)
    
    print(count)

solve()