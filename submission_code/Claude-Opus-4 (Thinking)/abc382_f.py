# Read input
H, W, N = map(int, input().split())
bars = []
for i in range(N):
    r, c, l = map(int, input().split())
    bars.append([r - 1, c - 1, l])  # Convert to 0-indexed

# Simulate the falling process
changed = True
while changed:
    changed = False
    
    # Build set of occupied cells once per iteration
    occupied = set()
    for j in range(N):
        rj, cj, lj = bars[j]
        for k in range(lj):
            occupied.add((rj, cj + k))
    
    # Process each bar in order
    for i in range(N):
        r, c, l = bars[i]
        
        # Check if bar can fall
        if r < H - 1:  # Not on bottom row
            # Check if all cells below are empty
            can_fall = True
            for j in range(l):
                if (r + 1, c + j) in occupied:
                    can_fall = False
                    break
            
            if can_fall:
                # Remove old positions from occupied set
                for j in range(l):
                    occupied.remove((r, c + j))
                
                # Move bar down
                bars[i][0] = r + 1
                changed = True
                
                # Add new positions to occupied set
                for j in range(l):
                    occupied.add((r + 1, c + j))

# Output final positions (convert back to 1-indexed)
for i in range(N):
    print(bars[i][0] + 1)