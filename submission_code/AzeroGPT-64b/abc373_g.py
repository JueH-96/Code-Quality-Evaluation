from typing import List
import sys
from math import atan2
input = sys.stdin.readline

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def check_intersect(p, q, rx, ry):
    x0, y0 = q[0] - p[0], q[1] - p[1]
    x1, y1 = ry[0] - rx[0], ry[1] - rx[1]
    # the most stupid way to decide intersect (But it works:D and got AC...)
    return cross((p[0]-rx[0], p[1]-rx[1]), (q[0]-ry[0], q[1]-ry[1])) * cross((q[0]-rx[0], q[1]-rx[1]), (p[0]-ry[0], p[1]-ry[1])) <= 0 and \\
        cross((rx[0]-p[0], rx[1]-p[1]), (ry[0]-q[0], ry[1]-q[1])) * cross((ry[0]-p[0], ry[1]-p[1]), (rx[0]-q[0], rx[1]-q[1])) <= 0


def main():
    N = int(input())
    ASP = [() for _ in range(N)]
    BDP = [() for _ in range(N)]
    for i in range(N):
        A, B = map(int, input().split())
        ASP[i] = (atan2(B, A), (A, B))
    for i in range(N):
        C, D = map(int, input().split())
        BDP[i] = (atan2(D, C), (C, D))

    pairs = []
    for i, (A_angle, A_point) in enumerate(sorted(ASP)):
        ASP[i] = A_point
        for j, (B_angle, B_point) in enumerate(sorted(BDP)):
            BDP[j] = B_point
            if A_angle <= B_angle:
                pairs.append((A_point, B_point))
                break
        else:
            pairs.append((A_point, BDP[0]))
            BDP[-1], BDP[0] = BDP[0], BDP[-1]

    result = [0]*N
    for i, ((A, B), (C, D)) in enumerate(pairs):
        inserted = False
        for j, (Ax, Bx, Cy, Dy) in enumerate(pairs):
            if check_intersect((A, B), (C, D), (Ax, Bx), (Cy, Dy)):
                p = pairs[:j] + [(A, B, C, D)] + pairs[j:]
                assert len(p) == N + 1
                p_j = min(p, key=lambda p: p[2])
                pairs = [p_j] + [p[i] for i in range(N + 1) if p[i] != p_j]
                inserted = True
                break
        if not inserted:
            pairs.append((A, B, C, D))

    for ((A, B), (C, D)) in pairs:
        for i, (A_, B_) in enumerate(ASP):
            if (A, B) == (A_, B_):
                a = i
        for j, (C_, D_) in enumerate(BDP):
            if (C, D) == (C_, D_):
                b = j
        result[a] = b+1

    if any(A == B for A, B in result):
        print(-1)
        return

    print(*result)


if __name__ == "__main__":
    main()