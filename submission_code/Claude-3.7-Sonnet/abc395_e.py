import heapq

def min_cost_to_reach_n(n, edges, x):
    # Create adjacency lists for original and reversed graphs
    original_graph = [[] for _ in range(n + 1)]
    reversed_graph = [[] for _ in range(n + 1)]
    
    for u, v in edges:
        original_graph[u].append(v)
        reversed_graph[v].append(u)
    
    # Priority queue: (cost, vertex, is_reversed)
    pq = [(0, 1, False)]  # Start at vertex 1 with 0 cost in original graph
    min_costs = {(1, False): 0}  # (vertex, is_reversed) -> min_cost
    
    while pq:
        cost, vertex, is_reversed = heapq.heappop(pq)
        
        if cost > min_costs[(vertex, is_reversed)]:
            continue
        
        if vertex == n:
            return cost  # Reached the destination
        
        # Choose the appropriate graph based on the current state
        current_graph = reversed_graph if is_reversed else original_graph
        
        # Move along edges
        for neighbor in current_graph[vertex]:
            new_cost = cost + 1
            if (neighbor, is_reversed) not in min_costs or new_cost < min_costs[(neighbor, is_reversed)]:
                min_costs[(neighbor, is_reversed)] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, is_reversed))
        
        # Reverse all edges
        new_cost = cost + x
        if (vertex, not is_reversed) not in min_costs or new_cost < min_costs[(vertex, not is_reversed)]:
            min_costs[(vertex, not is_reversed)] = new_cost
            heapq.heappush(pq, (new_cost, vertex, not is_reversed))
    
    return float('inf')  # Should not reach here if there's a path

# Read input
n, m, x = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# Find minimum cost
print(min_cost_to_reach_n(n, edges, x))