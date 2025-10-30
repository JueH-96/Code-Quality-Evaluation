def ccw(A, B, C):
    """Check if three points are in counter-clockwise order"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(P1, Q1, P2, Q2):
    """Check if line segments P1-Q1 and P2-Q2 intersect"""
    return ccw(P1, P2, Q2) != ccw(Q1, P2, Q2) and ccw(P1, Q1, P2) != ccw(P1, Q1, Q2)

def solve():
    N = int(input())
    P = []
    Q = []
    
    for i in range(N):
        a, b = map(int, input().split())
        P.append((a, b))
    
    for i in range(N):
        c, d = map(int, input().split())
        Q.append((c, d))
    
    # Try to find a valid permutation using angle-based sorting
    # First, find a central point (average of all points)
    all_points = P + Q
    cx = sum(p[0] for p in all_points) / len(all_points)
    cy = sum(p[1] for p in all_points) / len(all_points)
    
    import math
    
    # Sort P points by angle from center
    P_indexed = [(i, P[i]) for i in range(N)]
    P_indexed.sort(key=lambda x: math.atan2(x[1][1] - cy, x[1][0] - cx))
    
    # Sort Q points by angle from center
    Q_indexed = [(i, Q[i]) for i in range(N)]
    Q_indexed.sort(key=lambda x: math.atan2(x[1][1] - cy, x[1][0] - cx))
    
    # Create permutation by matching sorted orders
    R = [0] * N
    for i in range(N):
        P_idx = P_indexed[i][0]
        Q_idx = Q_indexed[i][0]
        R[P_idx] = Q_idx + 1  # 1-indexed
    
    # Verify no intersections
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if segments_intersect(P[i], Q[R[i] - 1], P[j], Q[R[j] - 1]):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        print(*R)
        return
    
    # If angle-based sorting doesn't work, try other approaches
    # Try sorting by x-coordinate
    P_indexed.sort(key=lambda x: (x[1][0], x[1][1]))
    Q_indexed.sort(key=lambda x: (x[1][0], x[1][1]))
    
    R = [0] * N
    for i in range(N):
        P_idx = P_indexed[i][0]
        Q_idx = Q_indexed[i][0]
        R[P_idx] = Q_idx + 1
    
    # Verify no intersections
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if segments_intersect(P[i], Q[R[i] - 1], P[j], Q[R[j] - 1]):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        print(*R)
    else:
        print(-1)

solve()