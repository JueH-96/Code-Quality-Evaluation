import heapq

def max_units(N, M, P):
    # Initialize heap with (marginal_cost, product_index, units_bought)
    heap = [(P[i], i, 0) for i in range(N)]
    heapq.heapify(heap)
    
    total_units = 0
    remaining_budget = M
    
    while heap:
        marginal_cost, idx, units = heapq.heappop(heap)
        
        # Check if we can afford this unit
        if marginal_cost <= remaining_budget:
            remaining_budget -= marginal_cost
            units += 1
            total_units += 1
            
            # Calculate and push next marginal cost
            next_marginal_cost = (2 * units + 1) * P[idx]
            heapq.heappush(heap, (next_marginal_cost, idx, units))
        else:
            # Can't afford the cheapest unit, so we're done
            break
    
    return total_units

# Read input
N, M = map(int, input().split())
P = list(map(int, input().split()))

# Output
print(max_units(N, M, P))