def solve():
    n, x, k = map(int, input().split())
    
    if k == 0:
        print(1)
        return
        
    distance = {}
    distance[x] = 0
    queue = [x]
    count_k_distance = 0
    
    while queue:
        u = queue.pop(0)
        if distance[u] == k:
            count_k_distance += 1
        if distance[u] >= k:
            continue
            
        parent = u // 2 if u > 1 else None
        child1 = 2 * u
        child2 = 2 * u + 1
        
        neighbors = []
        if parent is not None:
            neighbors.append(parent)
        if child1 <= n:
            neighbors.append(child1)
        if child2 <= n:
            neighbors.append(child2)
            
        for v in neighbors:
            if v not in distance and v <= n:
                distance[v] = distance[u] + 1
                queue.append(v)
                
    print(count_k_distance)

t = int(input())
for _ in range(t):
    solve()