def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) <= k:
                adj[i].append(j)
                
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            component = []
            q = [i]
            visited[i] = True
            component.append(i)
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        component.append(v)
                        q.append(v)
            components.append(component)
            
    possible = True
    for component in components:
        initial_values = set()
        target_values = set()
        for index in component:
            initial_values.add(a[index])
            target_values.add(b[index])
            
        for val in target_values:
            if val not in initial_values:
                possible = False
                break
        if not possible:
            break
            
    if possible:
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()