from collections import deque

def apply_operation(positions, i):
    # i is 1-indexed (1 to N-3)
    n = len(positions)
    
    # Create list of (position, original_index) and sort by position
    sorted_with_idx = sorted((pos, idx) for idx, pos in enumerate(positions))
    
    # Extract the indices of pieces that will move
    orig_idx_i1 = sorted_with_idx[i][1]
    orig_idx_i2 = sorted_with_idx[i+1][1]
    
    # Calculate new positions using the endpoints
    pos_i = sorted_with_idx[i-1][0]
    pos_i3 = sorted_with_idx[i+2][0]
    
    # Create new positions array
    new_positions = list(positions)
    new_positions[orig_idx_i1] = pos_i + pos_i3 - positions[orig_idx_i1]
    new_positions[orig_idx_i2] = pos_i + pos_i3 - positions[orig_idx_i2]
    
    return new_positions

def solve(n, positions):
    # Use BFS to find minimum sum
    initial = tuple(positions)
    visited = set()
    visited.add(initial)
    queue = deque([initial])
    min_sum = sum(positions)
    
    while queue:
        current = queue.popleft()
        current_sum = sum(current)
        min_sum = min(min_sum, current_sum)
        
        # Try all possible operations
        for i in range(1, n-2):  # i from 1 to N-3
            new_positions = apply_operation(list(current), i)
            new_tuple = tuple(new_positions)
            
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_sum = sum(new_positions)
                if new_sum <= min_sum:
                    queue.append(new_tuple)
    
    return min_sum

# Read input
n = int(input())
positions = list(map(int, input().split()))

# Solve and print result
print(solve(n, positions))