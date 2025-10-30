import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline

    N = int(input())
    pts = []
    # Read P points
    for i in range(1, N+1):
        x,y = map(int, input().split())
        pts.append((x,y,'P', i))
    # Read Q points
    for j in range(1, N+1):
        x,y = map(int, input().split())
        pts.append((x,y,'Q', j))

    # orientation of triplet a,b,c
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    # convex hull (monotone chain), returns list of points on hull in CCW order
    def convex_hull(ps):
        ps = sorted(ps, key=lambda p:(p[0],p[1]))
        if len(ps)<=1: return ps[:]
        lower = []
        for p in ps:
            while len(lower)>=2 and orient(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(ps):
            while len(upper)>=2 and orient(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        # drop last of each to avoid duplication
        return lower[:-1] + upper[:-1]

    # result mapping P_i -> Q_j
    match = dict()

    def solve(subpts):
        m = len(subpts)//2
        if m==0:
            return
        # find convex hull
        hull = convex_hull(subpts)
        L = len(hull)
        # find adjacent P-Q pair on hull
        for i in range(L):
            a = hull[i]
            b = hull[(i+1)%L]
            if a[2] != b[2]:
                # ensure a is P, b is Q, else swap
                if a[2]=='Q':
                    a,b = b,a
                # match P_i to Q_j
                _,_,_, pi = a
                _,_,_, qj = b
                match[pi] = qj
                # partition rest
                A = []
                B = []
                for p in subpts:
                    if p is a or p is b:
                        continue
                    val = orient(a,b,p)
                    if val>0:
                        A.append(p)
                    else:
                        B.append(p)
                # solve recursively
                solve(A)
                solve(B)
                return
        # should never reach here if solution exists
        print(-1)
        sys.exit(0)

    solve(pts)
    # output
    res = [0]*N
    for i in range(1, N+1):
        res[i-1] = match[i]
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()