from operator import itemgetter
from math import atan2, pi

def cross(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def ccw(p1, p2, p3):
    if cross(p1, p2, p3) > 0:
        return 1
    elif cross(p1, p2, p3) < 0:
        return -1
    elif (p2[0] - p1[0]) * (p3[0] - p1[0]) < 0:
        return 1
    elif (p2[1] - p1[1]) * (p3[1] - p1[1]) < 0:
        return 1
    elif (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 < (p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2:
        return 1
    else:
        return -1

def intersect(p1, p2, p3, p4):
    return ccw(p1, p2, p3) * ccw(p1, p2, p4) <= 0 and ccw(p3, p4, p1) * ccw(p3, p4, p2) <= 0

def solve(N, Ps, Qs):
    Qs = sorted(Qs, key=lambda q: atan2(q[1] - Ps[0][1], q[0] - Ps[0][0]))
    for i in range(N):
        Qs.append(Qs[i])
    for i in range(N):
        Qs[i] = (Qs[i], i)
    Qs = sorted(Qs, key=lambda q: (q[0][0] - Ps[0][0]) ** 2 + (q[0][1] - Ps[0][1]) ** 2)
    ans = [0] * N
    for i in range(2 * N):
        q, j = Qs[i][0], Qs[i][1]
        if ans[j] != 0:
            continue
        for k in range(N):
            if ans[k] == 0 and not intersect(Ps[k], q, Ps[0], Qs[i - 1][0]):
                ans[j] = k + 1
                break
    if 0 in ans:
        return [-1]
    return ans

N = int(input())
Ps = [tuple(map(int, input().split())) for _ in range(N)]
Qs = [tuple(map(int, input().split())) for _ in range(N)]
print(*solve(N, Ps, Qs))