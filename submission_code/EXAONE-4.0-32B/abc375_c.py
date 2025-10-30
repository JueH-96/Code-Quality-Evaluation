def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    grid = []
    for i in range(1, 1 + n):
        grid.append(list(data[i].strip()))
    
    res = [['' for _ in range(n)] for __ in range(n)]
    
    for i in range(n):
        for j in range(n):
            L = min(i + 1, j + 1, n - i, n - j)
            mod = L % 4
            if mod == 0:
                ni, nj = i, j
            elif mod == 1:
                ni, nj = j, n - 1 - i
            elif mod == 2:
                ni, nj = n - 1 - i, n - 1 - j
            else:  # mod == 3
                ni, nj = n - 1 - j, i
            res[ni][nj] = grid[i][j]
    
    for row in res:
        print(''.join(row))

if __name__ == '__main__':
    main()