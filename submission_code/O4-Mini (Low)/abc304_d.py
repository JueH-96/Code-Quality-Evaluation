import sys
import threading

def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it))
    H = int(next(it))
    N = int(next(it))
    pts = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        pts.append((x,y))
    A = int(next(it))
    a = [int(next(it)) for _ in range(A)]
    B = int(next(it))
    b = [int(next(it)) for _ in range(B)]
    # a and b are already sorted by problem statement

    # Count strawberries in each cell
    # cell indices: x-slot = number of cuts to the left => bisect_left on a
    # y-slot similarly.
    # Total cells = (A+1)*(B+1)
    from collections import defaultdict
    cnt = defaultdict(int)
    for x,y in pts:
        ix = bisect_left(a, x)
        iy = bisect_left(b, y)
        cnt[(ix,iy)] += 1

    total_cells = (A+1) * (B+1)
    # Maximum is just the max value in cnt (or 0 if no strawberries)
    M = 0
    if cnt:
        M = max(cnt.values())
    # Minimum: if there is any empty cell => 0, otherwise min value in cnt
    if len(cnt) < total_cells:
        m = 0
    else:
        m = min(cnt.values())

    print(m, M)

if __name__ == "__main__":
    main()