def min_cost_to_distribute_items(N, A, W):
    from collections import defaultdict
    
    # Step 1: Create a mapping of boxes to their items and their weights
    box_to_items = defaultdict(list)
    for i in range(N):
        box_to_items[A[i]].append(W[i])
    
    # Step 2: Calculate the cost to move items to make each box contain exactly one item
    total_cost = 0
    excess_items = []
    
    # Step 3: Identify boxes with more than one item
    for box, weights in box_to_items.items():
        if len(weights) > 1:
            # Sort weights to get the cheapest items to move out
            weights.sort()
            # Keep the cheapest item in the box and consider the rest as excess
            excess_items.extend(weights[:-1])  # all but the last (heaviest)
    
    # Step 4: Calculate the total cost of moving excess items to empty boxes
    # The number of empty boxes is equal to the number of boxes minus the number of boxes with items
    empty_boxes_count = N - len(box_to_items)
    
    # Sort excess items by weight to minimize cost
    excess_items.sort()
    
    # We can only move as many excess items as there are empty boxes
    for i in range(min(empty_boxes_count, len(excess_items))):
        total_cost += excess_items[i]
    
    return total_cost

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
A = list(map(int, data[1].split()))
W = list(map(int, data[2].split()))

# Calculate the minimum cost
result = min_cost_to_distribute_items(N, A, W)

# Print the result
print(result)