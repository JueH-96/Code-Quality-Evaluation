def solve():
    H, W, X, Y = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input())
    T = input()

    x, y = X, Y
    visited_houses = set()

    if grid[x-1][y-1] == '@':
        visited_houses.add((x, y))

    for move in T:
        nx, ny = x, y
        if move == 'U' and grid[x-2][y-1] != '#':
            nx = x - 1
        elif move == 'D' and grid[x][y-1] != '#':
            nx = x + 1
        elif move == 'L' and grid[x-1][y-2] != '#':
            ny = y - 1
        elif move == 'R' and grid[x-1][y] != '#':
            ny = y + 1
        
        if (nx, ny) != (x, y):
            x, y = nx, ny
            if grid[x-1][y-1] == '@':
                visited_houses.add((x, y))

    print(x, y, len(visited_houses))

solve()