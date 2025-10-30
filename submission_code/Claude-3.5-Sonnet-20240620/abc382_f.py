# YOUR CODE HERE
def simulate_bars(H, W, N, bars):
    # Create a grid to track occupied cells
    grid = [[False] * W for _ in range(H)]
    
    # Place initial bars
    for i, (R, C, L) in enumerate(bars):
        for j in range(L):
            grid[R-1][C-1+j] = True
    
    # Simulate bar movements
    changed = True
    while changed:
        changed = False
        for i, (R, C, L) in enumerate(bars):
            if R < H and all(not grid[R][C-1+j] for j in range(L)):
                # Move bar down
                for j in range(L):
                    grid[R-1][C-1+j] = False
                    grid[R][C-1+j] = True
                bars[i] = (R+1, C, L)
                changed = True
    
    # Return final positions
    return [R for R, _, _ in bars]

# Read input
H, W, N = map(int, input().split())
bars = [tuple(map(int, input().split())) for _ in range(N)]

# Simulate and print results
results = simulate_bars(H, W, N, bars)
for result in results:
    print(result)