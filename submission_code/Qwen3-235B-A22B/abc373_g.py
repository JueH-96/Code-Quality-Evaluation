import sys

def readints():
    return list(map(int, sys.stdin.readline().split()))

def is_clockwise(a, b, c):
    # Returns >0 if counter-clockwise, <0 if clockwise, 0 if colinear
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])

def segments_intersect(seg1, seg2):
    a1, a2 = seg1
    b1, b2 = seg2
    c1 = is_clockwise(a1, a2, b1)
    c2 = is_clockwise(a1, a2, b2)
    c3 = is_clockwise(b1, b2, a1)
    c4 = is_clockwise(b1, b2, a2)
    if c1 * c2 > 0 or c3 * c4 > 0:
        return False
    # Check if it's a proper intersection or just touching endpoints
    # We allow touching endpoints as not crossing
    if c1 == 0 or c2 == 0 or c3 == 0 or c4 == 0:
        return False
    return True

def main():
    N = int(sys.stdin.readline())
    P = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    Q = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Sort P and Q by x, then y
    sorted_p_indices = sorted(range(N), key=lambda i: (P[i][0], P[i][1]))
    sorted_q_indices = sorted(range(N), key=lambda i: (Q[i][0], Q[i][1]))
    
    R = [0] * N
    for i in range(N):
        p_idx = sorted_p_indices[i]
        q_idx = sorted_q_indices[i]
        R[p_idx] = q_idx + 1  # Convert to 1-based index
    
    # Output the result
    print(' '.join(map(str, R)))

main()