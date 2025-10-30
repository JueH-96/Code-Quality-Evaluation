from bisect import bisect_left, bisect_right
N = int(input())
D = [tuple(map(int, input().split())) for _ in range(N)]
CL = []
CR = []
CD = []
CU = []
for i in range(N):
    c, k = (D[i][0] + D[i][1], D[i][0] - D[i][1]) if D[i][0] >= D[i][1] else (D[i][1] + D[i][0], D[i][1] - D[i][0])
    CL.append(c)
    CR.append(-c)
    CD.append(k)
    CU.append(-k)
CL.sort()
CR.sort()
CD.sort()
CU.sort()
A = [(D[i][0] + D[i][1], 0) for i in range(N)]
B = [(D[i][1] + D[i][0], 1) for i in range(N)]
C = [(D[i][0] - D[i][1], 0) for i in range(N)]
D = [(D[i][1] - D[i][0], 1) for i in range(N)]
L = A + B
L.sort()
R = C + D
R.sort()
ans = 0
for c, k in L:
    c = (c * (1 - k) + (-c) * k)
    l = CL[bisect_left(CL, c) - 1] - c
    r = c - CR[bisect_right(CR, -c) - 1]
    ans += l + r
for c, k in R:
    c = (c * (1 - k) + (-c) * k)
    l = CD[bisect_left(CD, c) - 1] - c
    r = c - CU[bisect_right(CU, -c) - 1]
    ans += l + r
print(ans // 2)