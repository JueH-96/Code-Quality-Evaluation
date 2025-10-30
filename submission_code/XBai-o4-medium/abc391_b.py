n, m = map(int, input().split())
s_grid = [input().strip() for _ in range(n)]
t_grid = [input().strip() for _ in range(m)]

for a in range(1, n - m + 2):
    for b in range(1, n - m + 2):
        match = True
        for i in range(m):
            for j in range(m):
                if s_grid[a-1 + i][b-1 + j] != t_grid[i][j]:
                    match = False
                    break
            if not match:
                break
        if match:
            print(a, b)
            exit()