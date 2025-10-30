def main():
    import sys
    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    r, c = 0, 0
    dr, dc = -1, 0
    
    for _ in range(N):
        if grid[r][c] == '.':
            grid[r][c] = '#'
            dr, dc = dc, -dr
        else:
            grid[r][c] = '.'
            dr, dc = -dc, dr
        
        r = (r + dr) % H
        c = (c + dc) % W
    
    for row in grid:
        print(''.join(row))

if __name__ == '__main__':
    main()