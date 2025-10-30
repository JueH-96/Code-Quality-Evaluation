from collections import defaultdict

def min_cost_to_rearrange(N, A, W):
    # Create a dictionary to store the items in each box
    box_to_items = defaultdict(list)
    for i in range(N):
        box_to_items[A[i]].append((W[i], i+1))
    
    # Sort the items in each box by weight
    for box in box_to_items:
        box_to_items[box].sort()
    
    total_cost = 0
    for box in range(1, N+1):
        if box_to_items[box]:
            # Move the lightest item in the box to its correct position
            weight, item = box_to_items[box][0]
            if item != box:
                total_cost += weight
                # Remove the item from its current box and add it to the correct box
                box_to_items[A[item-1]].remove((weight, item))
                box_to_items[box].append((weight, item))
                box_to_items[box].sort()
    
    return total_cost

# Read input
N = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Solve the problem
print(min_cost_to_rearrange(N, A, W))