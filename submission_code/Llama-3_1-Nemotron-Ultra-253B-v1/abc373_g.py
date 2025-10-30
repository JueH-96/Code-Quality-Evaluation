import sys
import math
from functools import cmp_to_key

def readints():
    return list(map(int, sys.stdin.readline().split()))

def centroid(points):
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    return (sum_x / n, sum_y / n)

def compare(p1, p2, o):
    ax = p1[0] - o[0]
    ay = p1[1] - o[1]
    bx = p2[0] - o[0]
    by = p2[1] - o[1]
    cross = ax * by - ay * bx
    if cross > 1e-8:
        return -1
    elif cross < -1e-8:
        return 1
    else:
        dist1 = ax**2 + ay**2
        dist2 = bx**2 + by**2
        if dist1 < dist2:
            return -1
        elif dist1 > dist2:
            return 1
        else:
            return 0

def orientation(p, q, r):
    val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if val > 1e-8:
        return 1
    elif val < -1e-8:
        return -1
    else:
        return 0

def segments_intersect(a1, a2, b1, b2):
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    return False

def main():
    n = int(sys.stdin.readline())
    p = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    q = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    all_points = p + q
    o = centroid(all_points)
    p_sorted = sorted(p, key=cmp_to_key(lambda a, b: compare(a, b, o)))
    q_sorted = sorted(q, key=cmp_to_key(lambda a, b: compare(a, b, o)))
    R = []
    q_indices = []
    for pi in p_sorted:
        for i in range(n):
            if pi == p[i]:
                R.append(i+1)
                break
    q_order = []
    for qi in q_sorted:
        for i in range(n):
            if qi == q[i]:
                q_order.append(i+1)
                break
    R = q_order
    segments = []
    for i in range(n):
        a = p[i]
        b = q[R[i]-1]
        segments.append((a, b))
    valid = True
    for i in range(n):
        a1, a2 = segments[i]
        for j in range(i+1, n):
            b1, b2 = segments[j]
            if segments_intersect(a1, a2, b1, b2):
                valid = False
                break
        if not valid:
            break
    if valid:
        print(' '.join(map(str, R)))
    else:
        print(-1)

if __name__ == '__main__':
    main()