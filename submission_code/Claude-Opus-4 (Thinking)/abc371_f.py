n = int(input())
positions = list(map(int, input().split()))
q = int(input())

total_moves = 0

for _ in range(q):
    t, g = map(int, input().split())
    t -= 1  # Convert to 0-indexed
    
    old_positions = positions[:]
    new_positions = [0] * n
    
    # Place person t at position g
    new_positions[t] = g
    
    # Place persons to the left of t
    next_pos = g - 1
    for i in range(t - 1, -1, -1):
        new_positions[i] = min(positions[i], next_pos)
        next_pos = new_positions[i] - 1
    
    # Place persons to the right of t  
    next_pos = g + 1
    for i in range(t + 1, n):
        new_positions[i] = max(positions[i], next_pos)
        next_pos = new_positions[i] + 1
    
    # Calculate total moves
    for i in range(n):
        total_moves += abs(new_positions[i] - positions[i])
    
    positions = new_positions

print(total_moves)