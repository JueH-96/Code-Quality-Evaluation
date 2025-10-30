def can_reach_end(N, M, A, B, bad_ranges):
    # Convert bad ranges to list of tuples for easier processing
    bad_ranges = [(L, R) for L, R in bad_ranges]
    
    # Function to check if a position is bad
    def is_bad(pos):
        for L, R in bad_ranges:
            if L <= pos <= R:
                return True
        return False
    
    # Function to find next valid position
    def find_next_valid_positions(curr_pos):
        valid_positions = []
        for step in range(A, B + 1):
            next_pos = curr_pos + step
            if next_pos > N:
                break
            if not is_bad(next_pos):
                valid_positions.append(next_pos)
        return valid_positions
    
    # BFS to find if we can reach the end
    from collections import deque
    
    visited = set()
    queue = deque([1])  # Start from position 1
    visited.add(1)
    
    while queue:
        curr_pos = queue.popleft()
        
        # If we can reach N from current position in one step
        if curr_pos + A <= N <= curr_pos + B and not is_bad(N):
            return True
            
        # Try all possible next positions
        next_positions = find_next_valid_positions(curr_pos)
        for next_pos in next_positions:
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append(next_pos)
    
    return False

# Read input
N, M, A, B = map(int, input().split())
bad_ranges = []
for _ in range(M):
    L, R = map(int, input().split())
    bad_ranges.append((L, R))

# Print result
print("Yes" if can_reach_end(N, M, A, B, bad_ranges) else "No")