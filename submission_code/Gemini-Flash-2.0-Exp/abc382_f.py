def solve():
    H, W, N = map(int, input().split())
    bars = []
    for _ in range(N):
        bars.append(list(map(int, input().split())))

    grid = [[0] * W for _ in range(H)]
    for i in range(N):
        r, c, l = bars[i]
        for j in range(l):
            grid[r-1][c-1+j] = i + 1

    for _ in range(2 * N):
        for i in range(N):
            r, c, l = bars[i]
            
            can_move = True
            if r == H:
                can_move = False
            else:
                for j in range(l):
                    if r < H:
                        if grid[r][c-1+j] != 0:
                            can_move = False
                            break
                    else:
                        can_move = False
                        break
            
            if can_move:
                for j in range(l):
                    grid[r-1][c-1+j] = 0
                    grid[r][c-1+j] = i + 1
                bars[i][0] += 1
                

    for i in range(N):
        print(bars[i][0])

solve()