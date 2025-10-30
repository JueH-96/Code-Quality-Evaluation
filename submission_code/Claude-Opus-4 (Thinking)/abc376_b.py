# YOUR CODE HERE
def min_distance_avoiding(start, end, avoid, n):
    if start == end:
        return 0
    
    # Try clockwise path
    cw_dist = 0
    pos = start
    cw_valid = True
    while pos != end:
        pos = pos % n + 1
        cw_dist += 1
        if pos == avoid and pos != end:
            cw_valid = False
            break
    
    # Try counter-clockwise path
    ccw_dist = 0
    pos = start
    ccw_valid = True
    while pos != end:
        pos = n if pos == 1 else pos - 1
        ccw_dist += 1
        if pos == avoid and pos != end:
            ccw_valid = False
            break
    
    # Return minimum valid distance
    if cw_valid and ccw_valid:
        return min(cw_dist, ccw_dist)
    elif cw_valid:
        return cw_dist
    elif ccw_valid:
        return ccw_dist
    else:
        return float('inf')

# Read input
n, q = map(int, input().split())

# Initialize hand positions
left_pos = 1
right_pos = 2
total_ops = 0

# Process each instruction
for _ in range(q):
    hand, target = input().split()
    target = int(target)
    
    if hand == 'L':
        # Move left hand, avoiding right hand position
        ops = min_distance_avoiding(left_pos, target, right_pos, n)
        left_pos = target
    else:  # hand == 'R'
        # Move right hand, avoiding left hand position
        ops = min_distance_avoiding(right_pos, target, left_pos, n)
        right_pos = target
    
    total_ops += ops

print(total_ops)