from sys import stdin
from itertools import permutations

def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    """Find the orientation of an ordered triplet (p, q, r)."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or counterclockwise

def do_intersect(p1, q1, p2, q2):
    """Check if the line segment 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 != o2 and o3 != o4):
        return True

    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True

    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True

    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True

    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False

def solve():
    n = int(stdin.readline())
    p = []
    q = []
    for _ in range(n):
        x, y = map(int, stdin.readline().split())
        p.append((x, y))
    for _ in range(n):
        x, y = map(int, stdin.readline().split())
        q.append((x, y))

    for r in permutations(range(n)):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if do_intersect(p[i], q[r[i]], p[j], q[r[j]]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            print(' '.join(map(lambda x: str(x + 1), r)))
            return
    print(-1)

if __name__ == "__main__":
    solve()