H, W = map(int, input().split())
S = [list(input()) for _ in range(H)]
dx = [-1, 0, 1, 0, -1, 1, -1, 1]
dy = [0, 1, 0, -1, 1, 1, -1, -1]
snuke = list('snuke')

for i in range(H):
    for j in range(W):
        if S[i][j] == 's':
            for k in range(8):
                for l in range(1, 5):
                    ni = i + dx[k] * l
                    nj = j + dy[k] * l
                    if ni < 0 or H <= ni or nj < 0 or W <= nj or S[ni][nj] != snuke[l]:
                        break
                else:
                    for l in range(5):
                        print(i + dx[k] * l + 1, j + dy[k] * l + 1)
                    exit()