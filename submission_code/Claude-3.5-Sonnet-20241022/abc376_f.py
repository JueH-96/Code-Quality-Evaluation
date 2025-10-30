def get_min_moves(N, curr_left, curr_right, target_hand, target_pos):
    # BFS to find minimum moves to reach target position
    from collections import deque
    
    visited = set()
    queue = deque([(curr_left, curr_right, 0)])  # (left_pos, right_pos, moves)
    visited.add((curr_left, curr_right))
    
    while queue:
        left, right, moves = queue.popleft()
        
        # Check if we've reached target
        if (target_hand == 'L' and left == target_pos) or (target_hand == 'R' and right == target_pos):
            return moves, left, right
            
        # Try all possible moves
        # For left hand
        for next_left in [(left - 1) if left > 1 else N, (left + 1) if left < N else 1]:
            if next_left != right and (next_left, right) not in visited:
                visited.add((next_left, right))
                queue.append((next_left, right, moves + 1))
                
        # For right hand
        for next_right in [(right - 1) if right > 1 else N, (right + 1) if right < N else 1]:
            if next_right != left and (left, next_right) not in visited:
                visited.add((left, next_right))
                queue.append((left, next_right, moves + 1))
    
    return float('inf'), curr_left, curr_right  # Should never reach here given problem constraints

def main():
    # Read input
    N, Q = map(int, input().split())
    instructions = []
    for _ in range(Q):
        hand, pos = input().split()
        instructions.append((hand, int(pos)))
    
    # Process each instruction
    total_moves = 0
    curr_left, curr_right = 1, 2  # Initial positions
    
    for hand, target_pos in instructions:
        # If already at target position, no moves needed
        if (hand == 'L' and curr_left == target_pos) or (hand == 'R' and curr_right == target_pos):
            continue
            
        # Find minimum moves needed
        moves, curr_left, curr_right = get_min_moves(N, curr_left, curr_right, hand, target_pos)
        total_moves += moves
    
    print(total_moves)

if __name__ == "__main__":
    main()