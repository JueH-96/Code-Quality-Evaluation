import heapq

N, M = map(int, input().split())
P = list(map(int, input().split()))

# Use a priority queue to efficiently find the minimum cost increase
# Each element is (cost_increase, product_index, current_units)
heap = [(P[i], i, 0) for i in range(N)]
heapq.heapify(heap)

total_cost = 0
total_units = 0

while heap:
    cost_increase, idx, current_units = heapq.heappop(heap)
    
    if total_cost + cost_increase <= M:
        total_cost += cost_increase
        total_units += 1
        
        # Push the next cost increase for this product
        new_units = current_units + 1
        new_cost_increase = (2 * new_units + 1) * P[idx]
        heapq.heappush(heap, (new_cost_increase, idx, new_units))
    else:
        break

print(total_units)