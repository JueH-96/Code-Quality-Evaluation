def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    first_line = data[0].split()
    H = int(first_line[0])
    W = int(first_line[1])
    X = int(first_line[2])
    Y = int(first_line[3])
    
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    T = data[1+H].strip()
    
    x = X - 1
    y = Y - 1
    houses = set()
    
    if grid[x][y] == '@':
        houses.add((x, y))
        
    for move in T:
        dx, dy = 0, 0
        if move == 'U':
            dx = -1
        elif move == 'D':
            dx = 1
        elif move == 'L':
            dy = -1
        elif move == 'R':
            dy = 1
            
        nx = x + dx
        ny = y + dy
        
        if 0 <= nx < H and 0 <= ny < W:
            if grid[nx][ny] != '#':
                x, y = nx, ny
                
        if grid[x][y] == '@':
            houses.add((x, y))
            
    print(f"{x+1} {y+1} {len(houses)}")

if __name__ == "__main__":
    main()