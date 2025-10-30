# YOUR CODE HERE
def move(x, y, direction, grid):
    dx, dy = 0, 0
    if direction == 'U':
        dx = -1
    elif direction == 'D':
        dx = 1
    elif direction == 'L':
        dy = -1
    elif direction == 'R':
        dy = 1
    
    new_x, new_y = x + dx, y + dy
    if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] != '#':
        return new_x, new_y
    return x, y

def solve():
    H, W, X, Y = map(int, input().split())
    X -= 1  # Convert to 0-based indexing
    Y -= 1

    grid = [input().strip() for _ in range(H)]
    T = input().strip()

    houses = set()
    x, y = X, Y

    for move_dir in T:
        x, y = move(x, y, move_dir, grid)
        if grid[x][y] == '@':
            houses.add((x, y))

    return x + 1, y + 1, len(houses)  # Convert back to 1-based indexing for output

# Read input, solve the problem, and print the output
result = solve()
print(*result)