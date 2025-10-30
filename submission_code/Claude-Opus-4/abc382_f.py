# YOUR CODE HERE
H, W, N = map(int, input().split())

bars = []
for i in range(N):
    r, c, l = map(int, input().split())
    bars.append([r, c, l])  # row, column, length

# Simulate until stable
changed = True
while changed:
    changed = False
    
    # Create a grid to track occupied cells
    occupied = [[False] * (W + 1) for _ in range(H + 1)]
    
    # Mark all occupied cells
    for i in range(N):
        r, c, l = bars[i]
        for j in range(l):
            occupied[r][c + j] = True
    
    # Process each bar in order
    for i in range(N):
        r, c, l = bars[i]
        
        # Check if bar can move down
        if r < H:  # Not at bottom row
            can_move = True
            # Check all cells directly below
            for j in range(l):
                if occupied[r + 1][c + j]:
                    can_move = False
                    break
            
            if can_move:
                # Move the bar down
                bars[i][0] = r + 1
                changed = True
                
                # Update occupied grid
                for j in range(l):
                    occupied[r][c + j] = False
                    occupied[r + 1][c + j] = True

# Output final positions
for i in range(N):
    print(bars[i][0])