import sys
import heapq
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
W = list(map(int, data[N+1:2*N+1]))

# Mapping from box number to list of (weight, item_index)
box_to_items = defaultdict(list)

for i in range(N):
    box_to_items[A[i]].append((W[i], i + 1))

# Min-heap for the costs of moving items
min_cost_heap = []

# To store the total minimum cost
total_min_cost = 0

# We need to ensure each box from 1 to N has exactly one item
for box in range(1, N + 1):
    if box in box_to_items:
        # Sort items by weight in ascending order for each box
        items = sorted(box_to_items[box], key=lambda x: x[0])
        # The first item is the cheapest to keep, others need to be moved
        for weight, item_index in items[1:]:
            heapq.heappush(min_cost_heap, weight)
    else:
        # If there's no item in this box, we need to take the cheapest available item
        if min_cost_heap:
            total_min_cost += heapq.heappop(min_cost_heap)

# Now, we need to assign items to empty boxes
for box in range(1, N + 1):
    if not box in box_to_items:
        # We need to move the cheapest item available to this box
        if min_cost_heap:
            total_min_cost += heapq.heappop(min_cost_heap)

print(total_min_cost)