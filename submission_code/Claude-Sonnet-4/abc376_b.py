def min_distance_on_ring(start, end, n, avoid):
    """Calculate minimum distance from start to end on a ring of size n, avoiding a specific position"""
    if start == end:
        return 0
    
    # Try clockwise direction
    clockwise_dist = 0
    pos = start
    can_go_clockwise = True
    
    while pos != end:
        next_pos = pos + 1 if pos < n else 1
        if next_pos == avoid:
            can_go_clockwise = False
            break
        pos = next_pos
        clockwise_dist += 1
    
    # Try counterclockwise direction
    counterclockwise_dist = 0
    pos = start
    can_go_counterclockwise = True
    
    while pos != end:
        next_pos = pos - 1 if pos > 1 else n
        if next_pos == avoid:
            can_go_counterclockwise = False
            break
        pos = next_pos
        counterclockwise_dist += 1
    
    # Return the minimum valid distance
    if can_go_clockwise and can_go_counterclockwise:
        return min(clockwise_dist, counterclockwise_dist)
    elif can_go_clockwise:
        return clockwise_dist
    elif can_go_counterclockwise:
        return counterclockwise_dist
    else:
        # This shouldn't happen according to problem constraints
        return float('inf')

# Read input
n, q = map(int, input().split())

# Initial positions
left_pos = 1
right_pos = 2
total_operations = 0

# Process each instruction
for _ in range(q):
    hand, target = input().split()
    target = int(target)
    
    if hand == 'L':
        # Move left hand to target, avoiding right hand
        operations = min_distance_on_ring(left_pos, target, n, right_pos)
        total_operations += operations
        left_pos = target
    else:  # hand == 'R'
        # Move right hand to target, avoiding left hand
        operations = min_distance_on_ring(right_pos, target, n, left_pos)
        total_operations += operations
        right_pos = target

print(total_operations)