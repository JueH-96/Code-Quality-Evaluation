def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    grid = [list(line.strip()) for line in data[1:1+n]]
    
    # Collect the coordinates on the outer border in clockwise order starting from (0,0)
    coords = []
    # Top row: (0,0) to (0, n-1)
    for j in range(n):
        coords.append((0, j))
    # Right column: from (1, n-1) to (n-1, n-1)
    for i in range(1, n):
        coords.append((i, n-1))
    # Bottom row: from (n-1, n-2) down to (n-1, 0)
    for j in range(n-2, -1, -1):
        coords.append((n-1, j))
    # Left column: from (n-2, 0) up to (1, 0)
    for i in range(n-2, 0, -1):
        coords.append((i, 0))
    
    # Extract the outer values
    outer_values = [grid[i][j] for (i, j) in coords]
    
    # Rotate clockwise by one: new first element becomes old last, new[i] = old[i-1]
    rotated = [outer_values[-1]] + outer_values[:-1]
    
    # Place the rotated values back into the grid
    for idx, (i, j) in enumerate(coords):
        grid[i][j] = rotated[idx]
    
    # Print the resulting grid
    for row in grid:
        print("".join(row))
        
if __name__ == '__main__':
    main()