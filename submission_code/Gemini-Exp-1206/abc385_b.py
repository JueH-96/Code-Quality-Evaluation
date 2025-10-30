def solve():
    h, w, x, y = map(int, input().split())
    grid = [input() for _ in range(h)]
    t = input()
    
    x -= 1
    y -= 1
    
    houses = set()
    
    if grid[x][y] == '@':
        houses.add((x, y))
    
    for move in t:
        nx, ny = x, y
        if move == 'U':
            nx -= 1
        elif move == 'D':
            nx += 1
        elif move == 'L':
            ny -= 1
        elif move == 'R':
            ny += 1
        
        if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] != '#':
            x, y = nx, ny
            if grid[x][y] == '@':
                houses.add((x, y))
    
    print(x + 1, y + 1, len(houses))

solve()