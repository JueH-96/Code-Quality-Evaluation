def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    adj = {}
    for prod in s:
        if prod[0] not in adj:
            adj[prod[0]] = set()
        adj[prod[0]].add(prod[1])
    
    count = 0
    visited = set()
    for start_node in adj:
        if start_node in visited:
            continue
        count += 1
        
        q = [start_node]
        visited.add(start_node)
        
        while q:
            curr = q.pop(0)
            if curr in adj:
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
    
    print(count)

solve()