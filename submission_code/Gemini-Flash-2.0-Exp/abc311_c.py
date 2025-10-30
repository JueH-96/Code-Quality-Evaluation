def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    a = [x - 1 for x in a]
    
    for start_node in range(n):
        visited = [False] * n
        path = []
        
        curr = start_node
        
        while not visited[curr]:
            visited[curr] = True
            path.append(curr)
            curr = a[curr]
            
        if curr in path:
            cycle_start_index = path.index(curr)
            cycle = path[cycle_start_index:]
            
            print(len(cycle))
            print(*[x + 1 for x in cycle])
            return

solve()