def min_distance(n, from_pos, to_pos, avoid_pos):
    """Calculate minimum distance on a ring from from_pos to to_pos, avoiding avoid_pos"""
    if from_pos == to_pos:
        return 0
    
    # Try clockwise
    clockwise = 0
    curr = from_pos
    while curr != to_pos:
        curr = curr % n + 1
        clockwise += 1
        if curr == avoid_pos:
            clockwise = float('inf')
            break
    
    # Try counterclockwise
    counter_clockwise = 0
    curr = from_pos
    while curr != to_pos:
        curr = (curr - 2) % n + 1
        counter_clockwise += 1
        if curr == avoid_pos:
            counter_clockwise = float('inf')
            break
    
    return min(clockwise, counter_clockwise)

def solve():
    n, q = map(int, input().split())
    
    # Initial positions
    left_pos = 1
    right_pos = 2
    total_ops = 0
    
    for _ in range(q):
        hand, target = input().split()
        target = int(target)
        
        if hand == 'L':
            # Move left hand to target
            if left_pos != target:
                # Find minimum path avoiding right hand
                ops = min_distance(n, left_pos, target, right_pos)
                if ops == float('inf'):
                    # Need to move right hand first
                    # Find a position for right hand that allows left to reach target
                    min_total = float('inf')
                    for new_right in range(1, n + 1):
                        if new_right != left_pos and new_right != target:
                            # Cost to move right hand to new position
                            right_ops = min_distance(n, right_pos, new_right, left_pos)
                            # Cost to move left hand to target
                            left_ops = min_distance(n, left_pos, target, new_right)
                            if right_ops != float('inf') and left_ops != float('inf'):
                                min_total = min(min_total, right_ops + left_ops)
                    total_ops += min_total
                    right_pos = new_right
                else:
                    total_ops += ops
                left_pos = target
        else:  # hand == 'R'
            # Move right hand to target
            if right_pos != target:
                # Find minimum path avoiding left hand
                ops = min_distance(n, right_pos, target, left_pos)
                if ops == float('inf'):
                    # Need to move left hand first
                    # Find a position for left hand that allows right to reach target
                    min_total = float('inf')
                    for new_left in range(1, n + 1):
                        if new_left != right_pos and new_left != target:
                            # Cost to move left hand to new position
                            left_ops = min_distance(n, left_pos, new_left, right_pos)
                            # Cost to move right hand to target
                            right_ops = min_distance(n, right_pos, target, new_left)
                            if left_ops != float('inf') and right_ops != float('inf'):
                                min_total = min(min_total, left_ops + right_ops)
                    total_ops += min_total
                    left_pos = new_left
                else:
                    total_ops += ops
                right_pos = target
    
    print(total_ops)

solve()