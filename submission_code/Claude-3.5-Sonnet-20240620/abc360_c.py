# YOUR CODE HERE
def min_cost_to_arrange(N, A, W):
    # Create a list of tuples (item, box, weight)
    items = [(i+1, A[i], W[i]) for i in range(N)]
    
    # Sort items by their current box
    items.sort(key=lambda x: x[1])
    
    total_cost = 0
    i = 0
    while i < N:
        if items[i][1] == items[i][0]:  # Item is in the correct box
            i += 1
            continue
        
        cycle_start = i
        cycle_weight = float('inf')
        cycle_length = 0
        
        while True:
            cycle_weight = min(cycle_weight, items[i][2])
            next_box = items[i][0] - 1
            cycle_length += 1
            
            if next_box == cycle_start:
                break
            
            i = next_box
        
        total_cost += min(cycle_weight * (cycle_length - 1), cycle_weight * (cycle_length + 1))
        i = cycle_start + 1
    
    return total_cost

# Read input
N = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Calculate and print the result
result = min_cost_to_arrange(N, A, W)
print(result)