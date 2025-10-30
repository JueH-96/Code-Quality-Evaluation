def solve():
    n = int(input().strip())
    r = list(input().strip())
    c = list(input().strip())
    grid = [['.' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if r[i] == c[j]:
                grid[i][j] = r[i]
    for i in range(n):
        if grid[i].count('A') > 1 or grid[i].count('B') > 1 or grid[i].count('C') > 1:
            print('No')
            return
    for j in range(n):
        if [grid[i][j] for i in range(n)].count('A') > 1 or [grid[i][j] for i in range(n)].count('B') > 1 or [grid[i][j] for i in range(n)].count('C') > 1:
            print('No')
            return
    for i in range(n):
        if 'A' not in grid[i]:
            for j in range(n):
                if grid[j].count('A') == 0 and 'A' not in [grid[k][j] for k in range(n)]:
                    grid[i][j] = 'A'
                    break
        if 'B' not in grid[i]:
            for j in range(n):
                if grid[j].count('B') == 0 and 'B' not in [grid[k][j] for k in range(n)]:
                    grid[i][j] = 'B'
                    break
        if 'C' not in grid[i]:
            for j in range(n):
                if grid[j].count('C') == 0 and 'C' not in [grid[k][j] for k in range(n)]:
                    grid[i][j] = 'C'
                    break
    for i in range(n):
        if grid[i].count('A') != 1 or grid[i].count('B') != 1 or grid[i].count('C') != 1:
            print('No')
            return
    for j in range(n):
        if [grid[i][j] for i in range(n)].count('A') != 1 or [grid[i][j] for i in range(n)].count('B') != 1 or [grid[i][j] for i in range(n)].count('C') != 1:
            print('No')
            return
    print('Yes')
    for row in grid:
        print(''.join(row))

solve()