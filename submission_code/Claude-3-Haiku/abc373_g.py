import sys

def find_non_intersecting_segments():
    N = int(input())
    P = []
    Q = []
    for _ in range(N):
        A, B = map(int, input().split())
        P.append((A, B))
    for _ in range(N):
        C, D = map(int, input().split())
        Q.append((C, D))

    # Sort the points by their angle with the x-axis
    P.sort(key=lambda x: (x[0]**2 + x[1]**2)**0.5)
    Q.sort(key=lambda x: (x[0]**2 + x[1]**2)**0.5)

    # Find a permutation that satisfies the condition
    R = []
    used = [False] * N
    for i in range(N):
        for j in range(N):
            if not used[j]:
                # Check if the segment (P_i, Q_j) intersects with any previous segments
                intersects = False
                for k in range(i):
                    if check_intersection((P[i], Q[j]), (P[k], Q[R[k]])):
                        intersects = True
                        break
                if not intersects:
                    R.append(j+1)
                    used[j] = True
                    break

    if len(R) == N:
        return ' '.join(map(str, R))
    else:
        return '-1'

def check_intersection(seg1, seg2):
    # Check if two line segments intersect
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:
        return False
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d
    return 0 <= t <= 1 and 0 <= u <= 1

print(find_non_intersecting_segments())