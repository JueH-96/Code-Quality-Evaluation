import sys

def solve():
    N = int(sys.stdin.readline().strip())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().strip().split())
        points.append((x, y))
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().strip().split())
        points.append((x, y))

    # Check if there is a permutation of (1, 2, ..., N) that satisfies the condition
    for R in range(N):
        for S in range(R+1, N+1):
            # Check if the line segments connecting points P_R and Q_S intersect
            if not check_intersection(points[R], points[S]):
                print(R+1, S+1)
                return
    print(-1)

def check_intersection(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Check if the line segments P1P2 and Q1Q2 intersect
    if not check_intersection(p1, p2):
        return False
    # Check if the line segments P1Q1 and P2Q2 intersect
    if not check_intersection(p1, (x2, y2)):
        return False
    # Check if the line segments Q1P1 and Q2P2 intersect
    if not check_intersection((x1, y1), p2):
        return False
    return True

def check_intersection(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Check if the line P1P2 intersects the line P1Q1 and P2Q2
    if x1 == x2:
        return False
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y1 - y2) / (x1 - x2)
    if m1 == m2:
        return False
    b1 = y1 - m1 * x1
    b2 = y2 - m2 * x2
    if b1 == b2:
        return False
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    if (x, y) not in p1 and (x, y) not in p2:
        return False
    return True

if __name__ == "__main__":
    solve()