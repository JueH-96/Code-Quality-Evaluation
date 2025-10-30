def solve():
    n, x, k = map(int, input().split())
    
    def distance(a, b):
        dist = 0
        path_a = []
        path_b = []
        
        curr = a
        while curr > 0:
            path_a.append(curr)
            curr //= 2
        path_a = path_a[::-1]
        
        curr = b
        while curr > 0:
            path_b.append(curr)
            curr //= 2
        path_b = path_b[::-1]
        
        i = 0
        while i < min(len(path_a), len(path_b)) and path_a[i] == path_b[i]:
            i += 1
        
        dist = (len(path_a) - i) + (len(path_b) - i)
        return dist
    
    count = 0
    for i in range(1, min(n + 1, 2 * 10**5)):
        if distance(x, i) == k:
            count += 1
    
    if n > 2 * 10**5:
        
        def count_at_distance(start_node, dist, n):
            if dist == 0:
                return 1
            
            count = 0
            
            q = [(start_node, 0)]
            visited = {start_node}
            
            while q:
                node, d = q.pop(0)
                
                if d == dist:
                    count += 1
                elif d < dist:
                    neighbors = []
                    parent = node // 2
                    if parent > 0:
                        neighbors.append(parent)
                    
                    child1 = node * 2
                    child2 = node * 2 + 1
                    
                    if child1 <= n:
                        neighbors.append(child1)
                    if child2 <= n:
                        neighbors.append(child2)
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            q.append((neighbor, d + 1))
                            visited.add(neighbor)
            return count
        
        count = count_at_distance(x, k, n)
        
    print(count)

t = int(input())
for _ in range(t):
    solve()