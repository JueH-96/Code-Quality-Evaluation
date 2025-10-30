# YOUR CODE HERE
import sys
from collections import defaultdict

def min_cost_to_distribute_items(N, A, W):
    box_to_items = defaultdict(list)
    
    for i in range(N):
        box_to_items[A[i]].append(W[i])
    
    total_cost = 0
    for box, items in box_to_items.items():
        if len(items) > 1:
            items.sort()
            total_cost += sum(items[:-1])
    
    return total_cost

# Read input
input = sys.stdin.read
N, *A, *W = map(int, input().split())

# Solve the problem
result = min_cost_to_distribute_items(N, A, W)

# Print the result
print(result)