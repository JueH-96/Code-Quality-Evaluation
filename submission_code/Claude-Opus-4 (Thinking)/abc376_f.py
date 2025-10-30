from collections import deque

def get_adjacent(pos, N):
    """Get adjacent positions on a ring of size N"""
    prev_pos = pos - 1 if pos > 1 else N
    next_pos = pos + 1 if pos < N else 1
    return [prev_pos, next_pos]

def min_operations(N, left_pos, right_pos, hand, target):
    """
    Find minimum operations to move 'hand' to 'target' position.
    Returns: (operations, new_left_pos, new_right_pos)
    """
    if hand == 'L' and left_pos == target:
        return 0, left_pos, right_pos
    if hand == 'R' and right_pos == target:
        return 0, left_pos, right_pos
    
    # BFS
    queue = deque([(left_pos, right_pos, 0)])
    visited = {(left_pos, right_pos)}
    
    while queue:
        l, r, ops = queue.popleft()
        
        # Try moving left hand
        for next_l in get_adjacent(l, N):
            if next_l != r and (next_l, r) not in visited:
                visited.add((next_l, r))
                if hand == 'L' and next_l == target:
                    return ops + 1, next_l, r
                queue.append((next_l, r, ops + 1))
        
        # Try moving right hand
        for next_r in get_adjacent(r, N):
            if next_r != l and (l, next_r) not in visited:
                visited.add((l, next_r))
                if hand == 'R' and next_r == target:
                    return ops + 1, l, next_r
                queue.append((l, next_r, ops + 1))
    
    return -1, -1, -1

N, Q = map(int, input().split())

left_pos = 1
right_pos = 2
total_ops = 0

for _ in range(Q):
    parts = input().split()
    hand = parts[0]
    target = int(parts[1])
    
    ops, left_pos, right_pos = min_operations(N, left_pos, right_pos, hand, target)
    total_ops += ops

print(total_ops)