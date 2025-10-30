H, W = map(int, input().split())
S = [input() for _ in range(H)]
h = [False] * H
w = [False] * W
for i in range(H):
    for j in range(W):
        if S[i][j] == '#':
            h[i], w[j] = True, True
y = [False] * H
x = [False] * W
for i in range(H):
    if not all((h[i], h[i+1])):
        y[i] = True
    if not all((h[i], h[i-1])):
        y[i] = True
for i in range(W):
    if not all((w[i], w[i+1])):
        x[i] = True
    if not all((w[i], w[i-1])):
        x[i] = True
for i in range(H):
    for j in range(W):
        if S[i][j] == '.' and not x[j] and not y[i]:
            print((i+1, j+1))
            exit()