def solve():
    n = int(input())
    a = list(map(int, input().split()))
    w = list(map(int, input().split()))
    
    # Convert to 0-indexed
    for i in range(n):
        a[i] -= 1
    
    visited = [False] * n
    total_cost = 0
    
    for i in range(n):
        if visited[i] or a[i] == i:  # Already in correct position
            continue
            
        # Find cycle starting from position i
        cycle_weights = []
        current = i
        
        while not visited[current]:
            visited[current] = True
            cycle_weights.append(w[current])
            current = a[current]
        
        # For this cycle, minimum cost is sum - max
        if len(cycle_weights) > 1:
            total_cost += sum(cycle_weights) - max(cycle_weights)
    
    print(total_cost)

solve()