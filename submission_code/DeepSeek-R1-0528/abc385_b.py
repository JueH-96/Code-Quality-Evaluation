def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W, X, Y = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    T = data[1+H].strip()
    
    x = X - 1
    y = Y - 1
    houses = set()
    
    for c in T:
        nx, ny = x, y
        if c == 'U':
            nx = x - 1
        elif c == 'D':
            nx = x + 1
        elif c == 'L':
            ny = y - 1
        elif c == 'R':
            ny = y + 1
        
        if grid[nx][ny] != '#':
            x, y = nx, ny
        
        if grid[x][y] == '@':
            houses.add((x, y))
            
    print(f"{x+1} {y+1} {len(houses)}")

if __name__ == "__main__":
    main()