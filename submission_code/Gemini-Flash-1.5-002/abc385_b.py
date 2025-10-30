H, W, X, Y = map(int, input().split())
grid = [input() for _ in range(H)]
T = input()

X -= 1
Y -= 1

visited_houses = set()

for move in T:
    if move == 'U':
        if X > 0 and grid[X - 1][Y] != '#':
            X -= 1
            if grid[X][Y] == '@':
                visited_houses.add((X,Y))
    elif move == 'D':
        if X < H - 1 and grid[X + 1][Y] != '#':
            X += 1
            if grid[X][Y] == '@':
                visited_houses.add((X,Y))
    elif move == 'L':
        if Y > 0 and grid[X][Y - 1] != '#':
            Y -= 1
            if grid[X][Y] == '@':
                visited_houses.add((X,Y))
    elif move == 'R':
        if Y < W - 1 and grid[X][Y + 1] != '#':
            Y += 1
            if grid[X][Y] == '@':
                visited_houses.add((X,Y))

print(X + 1, Y + 1, len(visited_houses))