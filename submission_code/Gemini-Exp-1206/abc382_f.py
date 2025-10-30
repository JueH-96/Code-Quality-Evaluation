def solve():
    h, w, n = map(int, input().split())
    bars = []
    for _ in range(n):
        bars.append(list(map(int, input().split())))

    grid = {}
    for i in range(n):
        for j in range(bars[i][2]):
            grid[(bars[i][0], bars[i][1] + j)] = i

    for _ in range(min(h, 2 * n)):
        moved = [False] * n
        for i in range(n):
            r, c, l = bars[i]
            if r < h:
                can_move = True
                for j in range(l):
                    if (r + 1, c + j) in grid:
                        can_move = False
                        break
                if can_move:
                    for j in range(l):
                        del grid[(r, c + j)]
                    for j in range(l):
                        grid[(r + 1, c + j)] = i
                    bars[i][0] += 1
                    moved[i] = True
    
    for bar in bars:
        print(bar[0])

solve()