def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Group positions by value
    pos = {}
    for i, val in enumerate(a):
        if val not in pos:
            pos[val] = []
        pos[val].append(i)
    
    # Calculate minimum operations
    # For each value, calculate cost to collect all instances at front
    total_cost = 0
    processed = set()
    
    # Process values in order of first appearance
    values = []
    seen = set()
    for val in a:
        if val not in seen:
            values.append(val)
            seen.add(val)
    
    remaining_elements = list(range(n))
    
    for val in values:
        # Find positions of this value in remaining elements
        val_positions = []
        for i, orig_pos in enumerate(remaining_elements):
            if a[orig_pos] == val:
                val_positions.append(i)
        
        # Cost is the number of swaps needed to bring all to front
        swaps = 0
        for i, pos in enumerate(val_positions):
            swaps += pos - i
        
        total_cost += swaps + 1  # +1 for deletion
        
        # Remove these positions from remaining
        remaining_elements = [pos for i, pos in enumerate(remaining_elements) 
                            if i not in val_positions]
    
    return total_cost

t = int(input())
for _ in range(t):
    print(solve())