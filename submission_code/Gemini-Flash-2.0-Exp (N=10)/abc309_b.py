def solve():
    n = int(input())
    grid = []
    for _ in range(n):
        grid.append(list(map(int, list(input()))))
    
    if n <= 1:
      for row in grid:
        print("".join(map(str, row)))
      return

    outer_squares = []
    for j in range(n):
        outer_squares.append((0, j))
    for i in range(1, n):
        outer_squares.append((i, n - 1))
    for j in range(n - 2, -1, -1):
        outer_squares.append((n - 1, j))
    for i in range(n - 2, 0, -1):
        outer_squares.append((i, 0))
    
    shifted_values = []
    for r, c in outer_squares:
        shifted_values.append(grid[r][c])
    
    shifted_values = [shifted_values[-1]] + shifted_values[:-1]
    
    for i, (r, c) in enumerate(outer_squares):
        grid[r][c] = shifted_values[i]
    
    for row in grid:
        print("".join(map(str, row)))

solve()