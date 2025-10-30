def find_min_cost(N, A, W):
    # Create a graph where each node represents a box
    # and edges represent items that need to be moved
    graph = [[] for _ in range(N+1)]
    box_to_item = [0] * (N+1)
    
    # Map which item is in which box
    for i in range(N):
        box_to_item[A[i]] = i+1
    
    # For each box, find which item needs to be there
    for i in range(1, N+1):
        if box_to_item[i] != i:
            graph[i].append(box_to_item[i])
    
    # Find cycles in the graph
    visited = [False] * (N+1)
    cycles = []
    
    def find_cycle(start):
        cycle = []
        curr = start
        while not visited[curr]:
            visited[curr] = True
            cycle.append(curr)
            if not graph[curr]:
                return []
            curr = graph[curr][0]
            if curr == start:
                return cycle
        return []
    
    # Get all cycles
    for i in range(1, N+1):
        if not visited[i] and graph[i]:
            cycle = find_cycle(i)
            if cycle:
                cycles.append(cycle)
    
    total_cost = 0
    # For each cycle, find minimum weight item and rotate
    for cycle in cycles:
        if len(cycle) <= 1:
            continue
            
        # Find minimum weight in cycle
        min_weight = float('inf')
        for box in cycle:
            item = box_to_item[box]
            min_weight = min(min_weight, W[item-1])
        
        # Cost is (len-1) * min_weight + sum of other weights
        cycle_cost = (len(cycle)-1) * min_weight
        for box in cycle:
            item = box_to_item[box]
            cycle_cost = min(cycle_cost, cycle_cost - min_weight + W[item-1])
            
        total_cost += cycle_cost
        
    return total_cost

# Read input
N = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Print result
print(find_min_cost(N, A, W))