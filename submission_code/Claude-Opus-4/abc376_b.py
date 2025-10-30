# YOUR CODE HERE
def min_distance(n, start, end, blocked):
    """Calculate minimum distance from start to end on a ring of size n, avoiding blocked position"""
    if start == end:
        return 0
    
    # Clockwise distance
    if start < end:
        clockwise = end - start
    else:
        clockwise = n - start + end
    
    # Counterclockwise distance
    counterclockwise = n - clockwise
    
    # Check if blocked position is on the clockwise path
    blocked_on_clockwise = False
    pos = start
    for _ in range(clockwise):
        pos = pos % n + 1
        if pos == blocked:
            blocked_on_clockwise = True
            break
    
    # Check if blocked position is on the counterclockwise path
    blocked_on_counterclockwise = False
    pos = start
    for _ in range(counterclockwise):
        pos = pos - 1
        if pos == 0:
            pos = n
        if pos == blocked:
            blocked_on_counterclockwise = True
            break
    
    # Choose the path that doesn't contain the blocked position
    if blocked_on_clockwise and not blocked_on_counterclockwise:
        return counterclockwise
    elif blocked_on_counterclockwise and not blocked_on_clockwise:
        return clockwise
    else:
        # If both or neither path contains blocked, choose the shorter one
        return min(clockwise, counterclockwise)

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
        # Move left hand to target, right hand stays at right_pos
        operations = min_distance(n, left_pos, target, right_pos)
        left_pos = target
    else:  # hand == 'R'
        # Move right hand to target, left hand stays at left_pos
        operations = min_distance(n, right_pos, target, left_pos)
        right_pos = target
    
    total_operations += operations

print(total_operations)