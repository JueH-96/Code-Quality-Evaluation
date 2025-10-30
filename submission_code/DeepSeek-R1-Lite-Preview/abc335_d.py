def main():
    import sys
    sys.setrecursionlimit(1000000)
    N = int(sys.stdin.read())
    center = (N // 2, N // 2)
    grid = [['' for _ in range(N)] for _ in range(N)]
    grid[center[0]][center[1]] = 'T'
    directions = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up
    dir_idx = 0
    r, c = 0, 0
    grid[r][c] = 1
    count = 1
    while count < N*N - 1:
        dr, dc = directions[dir_idx]
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == '':
            r, c = nr, nc
            count += 1
            grid[r][c] = count
        else:
            dir_idx = (dir_idx + 1) % 4
    for row in grid:
        output = []
        for cell in row:
            if cell == '':
                output.append('T')
            else:
                output.append(str(cell))
        print(' '.join(output))

if __name__ == '__main__':
    main()