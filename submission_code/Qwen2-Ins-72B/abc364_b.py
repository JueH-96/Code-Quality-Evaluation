H, W = map(int, input().split())
si, sj = map(int, input().split())
C = [input() for _ in range(H)]
X = input()

di = {'U': -1, 'D': 1, 'L': 0, 'R': 0}
dj = {'U': 0, 'D': 0, 'L': -1, 'R': 1}

i, j = si - 1, sj - 1
for x in X:
    ni, nj = i + di[x], j + dj[x]
    if 0 <= ni < H and 0 <= nj < W and C[ni][nj] == '.':
        i, j = ni, nj

print(i + 1, j + 1)