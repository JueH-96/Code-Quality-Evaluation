def solve():
    n = int(input())
    grid = []
    for _ in range(n):
        row_str = input()
        row = [int(char) for char in row_str]
        grid.append(row)
    
    if n <= 1:
        for i in range(n):
            print("".join(map(str, grid[i])))
        return
        
    outer_squares_coords = []
    # First row
    for j in range(n):
        outer_squares_coords.append((0, j))
    # Last column (excluding first and last row)
    for i in range(1, n - 1):
        outer_squares_coords.append((i, n - 1))
    if n > 1:
        outer_squares_coords.append((n-1, n-1))
    # Last row (excluding last and first column)
    for j in range(n - 2, -1, -1):
        outer_squares_coords.append((n - 1, j))
    # First column (excluding last and first row)
    for i in range(n - 2, 0, -1):
        outer_squares_coords.append((i, 0))

    if n == 2:
        outer_squares_coords = [(0,0), (0,1), (1,1), (1,0)]
    elif n == 3:
        outer_squares_coords = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)]
    elif n >= 4:
        outer_squares_coords = []
        # First row
        for j in range(n):
            outer_squares_coords.append((0, j))
        # Last column (excluding first and last row)
        for i in range(1, n - 1):
            outer_squares_coords.append((i, n - 1))
        # Last row
        for j in range(n - 1, -1, -1):
            outer_squares_coords.append((n - 1, j))
        # First column (excluding first and last row)
        for i in range(n - 2, 0, -1):
            outer_squares_coords.append((i, 0))

    outer_values = []
    for r, c in outer_squares_coords:
        outer_values.append(grid[r][c])
        
    shifted_outer_values = [0] * len(outer_values)
    for i in range(len(outer_values)):
        shifted_outer_values[i] = outer_values[(i - 1) % len(outer_values)]
        
    for i in range(len(outer_squares_coords)):
        r, c = outer_squares_coords[i]
        grid[r][c] = shifted_outer_values[i]
        
    for i in range(n):
        print("".join(map(str, grid[i])))

if __name__ == '__main__':
    solve()