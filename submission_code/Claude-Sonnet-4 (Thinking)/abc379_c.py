n, m = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

# Check if total stones equals n
if sum(a) != n:
    print(-1)
else:
    # Create list of (position, stones) and sort by position
    stones = [(x[i], a[i]) for i in range(m)]
    stones.sort()
    
    supply_prefix = 0
    total_operations = 0
    possible = True
    prev_pos = 0
    
    for i in range(m):
        pos, num_stones = stones[i]
        
        # Check feasibility for positions prev_pos+1 to pos-1
        # In this range, supply_prefix stays constant, need supply_prefix >= pos-1
        if pos > prev_pos + 1 and supply_prefix < pos - 1:
            possible = False
            break
        
        # Calculate operations for positions prev_pos+1 to pos-1
        if pos > prev_pos + 1:
            count = pos - 1 - prev_pos
            sum_positions = (prev_pos + 1 + pos - 1) * count // 2
            total_operations += supply_prefix * count - sum_positions
        
        # Update supply_prefix with stones at current position
        supply_prefix += num_stones
        
        # Check feasibility at current position
        if supply_prefix < pos:
            possible = False
            break
        
        # Add operations for current position (if not last cell)
        if pos < n:
            total_operations += supply_prefix - pos
        
        prev_pos = pos
    
    # Check feasibility for remaining positions
    if possible and prev_pos < n and supply_prefix < n:
        possible = False
    
    # Calculate operations for remaining positions prev_pos+1 to n-1
    if possible and prev_pos < n - 1:
        count = n - 1 - prev_pos
        sum_positions = (prev_pos + 1 + n - 1) * count // 2
        total_operations += supply_prefix * count - sum_positions
    
    if possible:
        print(total_operations)
    else:
        print(-1)