def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    grid = [list(data[i+1].strip()) for i in range(N)]
    
    # Collect the coordinates of the outer layer in clockwise order
    coords = []
    # top row
    for j in range(N):
        coords.append((0, j))
    # right column (excluding the top corner)
    for i in range(1, N):
        coords.append((i, N-1))
    # bottom row (excluding the bottom-right corner), right-to-left
    for j in range(N-2, -1, -1):
        coords.append((N-1, j))
    # left column (excluding both corners), bottom-to-top
    for i in range(N-2, 0, -1):
        coords.append((i, 0))
    
    # Extract the values on the border
    vals = [grid[i][j] for (i, j) in coords]
    # Rotate clockwise by one: last element moves to front
    rotated = [vals[-1]] + vals[:-1]
    
    # Write them back
    for (pos, v) in zip(coords, rotated):
        i, j = pos
        grid[i][j] = v
    
    # Print the resulting grid
    out = sys.stdout
    for row in grid:
        out.write("".join(row) + "
")

if __name__ == "__main__":
    main()