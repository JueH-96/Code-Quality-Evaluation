def solve():
    h, w, n = map(int, input().split())
    bars = []
    for _ in range(n):
        r, c, l = map(int, input().split())
        bars.append([r, c, l])

    grid = {}
    for i in range(n):
        r, c, l = bars[i]
        for j in range(l):
            grid[(r, c + j)] = i + 1

    
    while True:
        moved = False
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
                        grid[(r + 1, c + j)] = i + 1
                    bars[i][0] += 1
                    moved = True
        if not moved:
            break

    for bar in bars:
        print(bar[0])

solve()