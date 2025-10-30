def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    grid = [list(map(int, list(row))) for row in data[1:]]

    # Collect positions of the outer ring in clockwise order
    ring_positions = []
    # Top row (left to right)
    for c in range(N):
        ring_positions.append((0, c))
    # Right column (top to bottom, excluding corners)
    for r in range(1, N - 1):
        ring_positions.append((r, N - 1))
    # Bottom row (right to left)
    if N > 1:
        for c in range(N - 1, -1, -1):
            ring_positions.append((N - 1, c))
    # Left column (bottom to top, excluding corners)
    for r in range(N - 2, 0, -1):
        ring_positions.append((r, 0))
    
    # Extract the values in the outer ring
    ring_values = [grid[r][c] for (r, c) in ring_positions]
    
    # Shift the ring values clockwise by 1
    # That means the last element will move to the front
    ring_values = [ring_values[-1]] + ring_values[:-1]
    
    # Place the shifted values back into the grid
    for pos, val in zip(ring_positions, ring_values):
        r, c = pos
        grid[r][c] = val
    
    # Print the resulting grid
    for r in range(N):
        print("".join(map(str, grid[r])))

# Do not forget to call main()
main()