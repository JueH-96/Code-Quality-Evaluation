# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    for k in range(1, 5):
                        ni, nj = i + di * k, j + dj * k
                        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == 'n':
                            for l in range(1, 5):
                                nni, nnj = ni + di * l, nj + dj * l
                                if 0 <= nni < H and 0 <= nnj < W and grid[nni][nnj] == 'u':
                                    for m in range(1, 5):
                                        nnni, nnnj = nni + di * m, nnj + dj * m
                                        if 0 <= nnni < H and 0 <= nnnj < W and grid[nnni][nnnj] == 'k':
                                            for n in range(1, 5):
                                                nnnni, nnnnj = nnni + di * n, nnnj + dj * n
                                                if 0 <= nnnni < H and 0 <= nnnnj < W and grid[nnnni][nnnnj] == 'e':
                                                    print(i + 1, j + 1)
                                                    print(ni + 1, nj + 1)
                                                    print(nni + 1, nnj + 1)
                                                    print(nnni + 1, nnj + 1)
                                                    print(nnnni + 1, nnnnj + 1)
                                                    break
                                    break
                            break