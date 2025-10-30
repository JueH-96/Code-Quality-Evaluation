def simulate_bars(H, W, N, bars):
    # Convert bars to list of [row, col, length] for easier manipulation
    bars = [[r, c, l] for r, c, l in bars]
    
    # Keep simulating until no movement is possible
    while True:
        moved = False
        # Process each bar in order
        for i in range(N):
            # Skip if bar is at bottom row
            if bars[i][0] == H:
                continue
                
            # Check if can move down
            can_move = True
            row = bars[i][0]
            col = bars[i][1]
            length = bars[i][2]
            
            # Check each position below the current bar
            for j in range(length):
                # For each position, check if any other bar occupies the cell below
                for k in range(N):
                    if k == i:
                        continue
                    # If another bar occupies this position
                    if (bars[k][0] == row + 1 and 
                        bars[k][1] <= col + j < bars[k][1] + bars[k][2]):
                        can_move = False
                        break
                if not can_move:
                    break
            
            # If can move, move the bar down
            if can_move:
                bars[i][0] += 1
                moved = True
        
        # If no bar moved in this iteration, we're done
        if not moved:
            break
    
    # Return final row positions
    return [bar[0] for bar in bars]

# Read input
H, W, N = map(int, input().split())
bars = []
for _ in range(N):
    r, c, l = map(int, input().split())
    bars.append([r, c, l])

# Get final positions and print
final_positions = simulate_bars(H, W, N, bars)
for pos in final_positions:
    print(pos)