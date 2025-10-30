H, W, N = map(int, input().split())
bars = []
for i in range(N):
    R, C, L = map(int, input().split())
    bars.append([R, C, L])

grid = {}
for i in range(N):
    R, C, L = bars[i]
    for j in range(L):
        grid[(R, C + j)] = i + 1

while True:
    moved = False
    for i in range(N):
        R, C, L = bars[i]
        can_move = True
        for j in range(L):
            if R + 1 > H or (R + 1, C + j) in grid:
                can_move = False
                break
        if can_move:
            moved = True
            for j in range(L):
                grid.pop((R, C + j))
            bars[i][0] += 1
            R += 1
            for j in range(L):
                grid[(R, C + j)] = i + 1
    if not moved:
        break

for i in range(N):
    print(bars[i][0])