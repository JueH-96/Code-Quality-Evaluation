import random
from operator import itemgetter

#room
llx, urx, lly, ury = [], [], [], []
N = int(input())
for i in range(N):
    X, Y, Z, W = map(int, input().split())
    llx.append(X)
    urx.append(Y)
    lly.append(Z)
    ury.append(W)

A = min(llx)
B = max(urx)
C = min(lly)
D = max(ury)
S = [[0 for i in range(D + 2)] for j in range(B + 2)]
for i in range(N):
    y0 = lly[i]
    y1 = ury[i] + 1
    x0 = llx[i]
    x1 = urx[i] + 1
    
    S[y0][x0] += 1
    S[y0][x1] -= 1
    S[y1][x0] -= 1
    S[y1][x1] += 1

sx = [0 for i in range(B+2)]
ans = 0

for j in range(1, D + 1):
    for i in range(1, B + 1):
        sx[i] = sx[i - 1] + S[j][i]

    x0 = 0
    for i in range(B + 1):
        if sx[i] >= 1 : continue
        ans += (x0 - i) * (D - j + 1)
        x0 = i
#end

print(C * A * B - ans)