H, W = map(int, input().split())
S = [input() for _ in range(H)]

for i in range(H):
    for j in range(W):
        if S[i][j] == 's':
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    found = True
                    for k in range(5):
                        ni, nj = i + k * di, j + k * dj
                        if ni < 0 or ni >= H or nj < 0 or nj >= W:
                            found = False
                            break
                        if S[ni][nj] != 'snuke'[k]:
                            found = False
                            break
                    if found:
                        for k in range(5):
                            print(i + k * di + 1, j + k * dj + 1)
                        exit()