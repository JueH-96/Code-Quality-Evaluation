def min_distance(start, target, blocked, n):
    if start == target:
        return 0
    
    # Convert to 0-indexed for easier modular arithmetic
    start -= 1
    target -= 1
    blocked -= 1
    
    clockwise_dist = (target - start) % n
    counterclockwise_dist = (start - target) % n
    
    # Check if blocked position is on clockwise path
    if start < target:
        clockwise_blocked = start < blocked <= target
    elif start > target:
        clockwise_blocked = blocked > start or blocked <= target
    else:
        clockwise_blocked = False
    
    # Check if blocked position is on counterclockwise path
    if start > target:
        counterclockwise_blocked = target <= blocked < start
    elif start < target:
        counterclockwise_blocked = blocked < start or blocked >= target
    else:
        counterclockwise_blocked = False
    
    if not clockwise_blocked and not counterclockwise_blocked:
        return min(clockwise_dist, counterclockwise_dist)
    elif not clockwise_blocked:
        return clockwise_dist
    elif not counterclockwise_blocked:
        return counterclockwise_dist
    else:
        # This should not happen according to the problem statement
        return float('inf')

n, q = map(int, input().split())
left_pos = 1
right_pos = 2
total_ops = 0

for _ in range(q):
    hand, target = input().split()
    target = int(target)
    
    if hand == 'L':
        # Move left hand to target, right hand stays fixed
        ops = min_distance(left_pos, target, right_pos, n)
        left_pos = target
    else:  # hand == 'R'
        # Move right hand to target, left hand stays fixed
        ops = min_distance(right_pos, target, left_pos, n)
        right_pos = target
    
    total_ops += ops

print(total_ops)