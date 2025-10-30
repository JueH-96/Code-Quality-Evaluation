def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    reachable_counts = []
    for start_node in range(1, n + 1):
        reachable = set()
        reachable.add(start_node)
        
        current_node = start_node
        while True:
            next_node = a[current_node - 1]
            if next_node in reachable:
                break
            reachable.add(next_node)
            current_node = next_node
        
        reachable_counts.append(len(reachable))
    
    total_reachable_pairs = sum(reachable_counts)
    print(total_reachable_pairs)

solve()