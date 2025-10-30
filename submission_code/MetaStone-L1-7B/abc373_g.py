import sys

def readints():
    return list(map(int, sys.stdin.readline().split()))

def ccw(a, b, c):
    # Returns the cross product of (b - a) and (c - a)
    # If positive, a->b->c is counter-clockwise
    # If zero, colinear
    # If negative, clockwise
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def segments_intersect(a1, a2, b1, b2):
    # Check if line segments (a1,a2) and (b1,b2) intersect
    ccw1 = ccw(a1, a2, b1)
    ccw2 = ccw(a1, a2, b2)
    ccw3 = ccw(b1, b2, a1)
    ccw4 = ccw(b1, b2, a2)
    if (ccw1 == 0 and ccw2 == 0) or (ccw3 == 0 and ccw4 == 0):
        return True
    if (ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0):
        return True
    return False

def main():
    N = int(sys.stdin.readline())
    P = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    Q = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Sort P and Q by x, then y
    P.sort()
    Q.sort()
    
    # Create a list of all possible edges and check if they can be added
    dp = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            # Check if adding P[i-1] to Q[j-1] is possible
            # without crossing any previous edges in dp[i-1][j-1]
            if i == 1 or j == 1:
                dp[i][j] = 1
            else:
                option1 = dp[i-1][j]
                option2 = dp[i][j-1]
                option3 = dp[i-1][j-1]
                if not segments_intersect(P[i-1], Q[j-1], P[i-2], Q[j-2]):
                    option3 += 1
                dp[i][j] = max(option1, option2, option3)
    
    if dp[N][N] < N:
        print(-1)
        return
    
    # Now reconstruct the permutation R
    R = [0]*N
    i = N
    j = N
    while i > 0 and j > 0:
        if segments_intersect(P[i-1], Q[j-1], P[i-2], Q[j-2]):
            i -= 1
            j -= 1
        else:
            R[j-1] = i
            i -= 1
            j -= 1
    
    if len(R) != N:
        print(-1)
        return
    
    print(' '.join(map(str, R)))
    
if __name__ == '__main__':
    main()