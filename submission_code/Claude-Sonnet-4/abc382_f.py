H, W, N = map(int, input().split())
bars = []
for i in range(N):
    R, C, L = map(int, input().split())
    bars.append([R, C, L])  # [row, col, length]

while True:
    moved = False
    
    # Create occupied set for current state
    occupied = set()
    for i in range(N):
        r, c, l = bars[i]
        for j in range(l):
            occupied.add((r, c + j))
    
    for i in range(N):
        r, c, l = bars[i]
        
        # Check if bar can move down
        if r < H:  # Not on bottom row
            can_move = True
            # Check if all cells below are free
            for j in range(l):
                if (r + 1, c + j) in occupied:
                    can_move = False
                    break
            
            if can_move:
                # Remove current position from occupied
                for j in range(l):
                    occupied.discard((r, c + j))
                
                # Move bar down
                bars[i][0] = r + 1
                
                # Add new position to occupied
                for j in range(l):
                    occupied.add((r + 1, c + j))
                
                moved = True
    
    if not moved:
        break

# Output final positions
for i in range(N):
    print(bars[i][0])