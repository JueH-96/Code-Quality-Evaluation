def main():
    import sys
    data = sys.stdin.read().split()
    H, W, X, Y = map(int, data[:4])
    grid = data[4:4+H]
    T = data[4+H]
    # convert to 0-based
    x, y = X-1, Y-1
    visited = set()
    # movements
    moves = {'U':(-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}
    for c in T:
        dx, dy = moves[c]
        nx, ny = x+dx, y+dy
        # check passable
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
            x, y = nx, ny
            if grid[x][y] == '@':
                visited.add((x,y))
    # output 1-based position and count
    print(x+1, y+1, len(visited))

if __name__ == "__main__":
    main()