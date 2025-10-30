# Read input
H, W, N = map(int, input().split())
bars = []
for _ in range(N):
    R, C, L = map(int, input().split())
    bars.append((R, C, L))

# Initialize the current positions of the bars
current_positions = []
for R, C, L in bars:
    current_positions.append((R, C, L))

# Simulate until no bar can move
while True:
    moved = False
    
    # Calculate the occupied cells for this time step
    occupied = set()
    for R, C, L in current_positions:
        for j in range(L):
            occupied.add((R, C + j))
    
    # Process bars in order from 1 to N
    for i in range(N):
        R, C, L = current_positions[i]
        
        # Remove this bar from the occupied set temporarily
        for j in range(L):
            occupied.remove((R, C + j))
        
        # Check if the bar can move down
        can_move = R < H  # Not already at the bottom row
        
        if can_move:
            # Check if any cell below the bar is occupied
            for j in range(L):
                if (R + 1, C + j) in occupied:
                    can_move = False
                    break
        
        if can_move:
            # Move the bar down by one cell
            current_positions[i] = (R + 1, C, L)
            
            # Add the new positions to the occupied set
            for j in range(L):
                occupied.add((R + 1, C + j))
            
            moved = True
        else:
            # Add the original positions back to the occupied set
            for j in range(L):
                occupied.add((R, C + j))
    
    if not moved:
        break

# Print only the row positions
for R, C, L in current_positions:
    print(R)