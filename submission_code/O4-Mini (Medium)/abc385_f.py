import sys
import threading

def main():
    import sys
    data = sys.stdin
    n_line = data.readline().strip()
    if not n_line:
        print(-1)
        return
    n = int(n_line)
    pts = []
    for _ in range(n):
        xh = data.readline().split()
        x = float(xh[0])
        h = float(xh[1])
        pts.append((x, h))
    # sort by x ascending
    pts.sort(key=lambda p: p[0])
    # build lower convex hull of points (x, h)
    hull = []
    def cross(p1, p2, p3):
        # cross product of (p2-p1) x (p3-p2)
        return (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
    for p in pts:
        # for lower hull: remove last while turn is non-left (i.e. cross <= 0)
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    # compute maximum y-intercept among edges of hull
    max_b = float('-inf')
    for i in range(1, len(hull)):
        x1, h1 = hull[i-1]
        x2, h2 = hull[i]
        # intercept at x=0 of line through (x1,h1),(x2,h2)
        # b = (h1*x2 - h2*x1) / (x2 - x1)
        num = h1*x2 - h2*x1
        den = x2 - x1
        b = num/den
        if b > max_b:
            max_b = b
    # decide output
    # if no positive intercept required (max_b < 0), then even at h=0 all visible
    # else max_b>=0 is answer
    eps = 1e-12
    if max_b < -eps:
        # no blocking height needed
        print(-1)
    else:
        # if tiny negative, clamp to zero
        if abs(max_b) < eps:
            max_b = 0.0
        # print with enough precision
        # required 1e-9; we print 12 decimals
        print("{:.12f}".format(max_b))

if __name__ == "__main__":
    main()