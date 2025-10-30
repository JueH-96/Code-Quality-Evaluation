import sys
import threading

def main():
    import sys
    data = sys.stdin
    t = int(data.readline())
    out = []
    for _ in range(t):
        n_line = data.readline().strip()
        # Skip empty lines if any
        while n_line == "":
            n_line = data.readline().strip()
        n = int(n_line)
        A = list(map(int, data.readline().split()))
        # Build prefix sums S and F = -S
        F = [0] * (n + 1)
        s = 0
        for i, a in enumerate(A, start=1):
            s += a
            F[i] = -s
        # Compute upper convex hull of points (i, F[i]), i = 0..n
        # We'll store hull as list of (x, y)
        hull = []
        # cross product of OA->OB and OA->OC
        def cross(o, a, b):
            # (a.x - o.x)*(b.y - o.y) - (a.y - o.y)*(b.x - o.x)
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        for i in range(n+1):
            pt = (i, F[i])
            # maintain upper hull: we want to pop while last turn is 
            # "below" or collinear for upper hull => cross >= 0
            while len(hull) >= 2 and cross(hull[-2], hull[-1], pt) >= 0:
                hull.pop()
            hull.append(pt)
        # Now hull is the upper convex hull of (i,F[i])
        # Check that each segment has integer slope
        ok = True
        for k in range(len(hull)-1):
            x1, y1 = hull[k]
            x2, y2 = hull[k+1]
            dx = x2 - x1
            dy = y2 - y1
            # dy must be divisible by dx
            if dy % dx != 0:
                ok = False
                break
        out.append("Yes" if ok else "No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()