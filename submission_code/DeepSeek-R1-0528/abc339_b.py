def main():
    import sys
    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    cur_i, cur_j = 0, 0
    cur_d = 0
    
    for _ in range(N):
        if grid[cur_i][cur_j] == '.':
            grid[cur_i][cur_j] = '#'
            cur_d = (cur_d + 1) % 4
        else:
            grid[cur_i][cur_j] = '.'
            cur_d = (cur_d + 3) % 4
        
        di, dj = dirs[cur_d]
        cur_i = (cur_i + di + H) % H
        cur_j = (cur_j + dj + W) % W
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()