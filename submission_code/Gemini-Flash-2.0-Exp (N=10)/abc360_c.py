def solve():
    n = int(input())
    a = list(map(int, input().split()))
    w = list(map(int, input().split()))

    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        adj[a[i]].append((i + 1, w[i]))

    visited = [False] * (n + 1)
    total_cost = 0

    for i in range(1, n + 1):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                
                if len(adj[curr]) > 0:
                    next_item, weight = adj[curr][0]
                    cycle.append((curr, next_item, weight))
                    
                    found_next_box = False
                    for j in range(1, len(adj[curr])):
                        next_item2, weight2 = adj[curr][j]
                        if not visited[next_item2]:
                            next_item = next_item2
                            weight = weight2
                            found_next_box = True
                            break
                    if not found_next_box:
                        adj[curr].pop(0)
                    curr = next_item
                else:
                    break
            
            if len(cycle) > 0:
                min_weight = float('inf')
                cycle_sum = 0
                for _, _, weight in cycle:
                    min_weight = min(min_weight, weight)
                    cycle_sum += weight
                total_cost += cycle_sum - min_weight
    print(total_cost)

solve()