import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    pts = []
    for _ in range(N):
        x, h = map(int, input().split())
        pts.append((x, h))

    # If there's only one building, it's always visible.
    if N <= 1:
        print(-1.0)
        return

    # Build the upper convex hull of the points (x, h) in increasing x.
    # We maintain hull such that for any three consecutive A, B, C,
    # the turn A->B->C is a "left" turn: cross((B-A),(C-B)) > 0.
    def cross(ax, ay, bx, by, cx, cy):
        # cross of vectors AB and BC
        return (bx-ax)*(cy-by) - (by-ay)*(cx-bx)

    hull = []
    for x, h in pts:
        # pop while we make a non-left turn
        while len(hull) >= 2:
            ax, ay = hull[-2]
            bx, by = hull[-1]
            cx, cy = x, h
            if cross(ax, ay, bx, by, cx, cy) <= 0:
                hull.pop()
            else:
                break
        hull.append((x, h))

    # Now for each consecutive pair on the hull compute the blocking threshold:
    # T = (H_k * X_i - H_i * X_k) / (X_i - X_k)
    best = -1e30
    for i in range(len(hull)-1):
        xk, hk = hull[i]
        xi, hi = hull[i+1]
        # xi > xk guaranteed
        num = hk*xi - hi*xk
        den = xi - xk
        t = num/den
        if t > best:
            best = t

    # If the maximum threshold is negative, all buildings are visible even at h=0.
    if best < 0:
        print(-1.0)
    else:
        # Otherwise, that's the highest h from which not all are visible.
        print("{:.15f}".format(best))

if __name__ == "__main__":
    main()