import sys
import threading

def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    it = iter(data)
    W = int(next(it))
    H = int(next(it))
    N = int(next(it))
    points = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        points.append((x, y))
    A = int(next(it))
    a = [int(next(it)) for _ in range(A)]
    B = int(next(it))
    b = [int(next(it)) for _ in range(B)]

    # Count strawberries in each piece.
    # Find for each point which x-interval and y-interval it belongs to.
    # x-interval index = number of a_j < x  -> bisect_left(a, x)
    # y-interval index = number of b_j < y  -> bisect_left(b, y)
    counts = {}
    for x, y in points:
        xi = bisect.bisect_left(a, x)
        yi = bisect.bisect_left(b, y)
        key = (xi, yi)
        counts[key] = counts.get(key, 0) + 1

    # Maximum is the max over occupied cells, or 0 if no strawberries at all
    M = 0
    if counts:
        M = max(counts.values())

    # Total number of cells
    total_cells = (A + 1) * (B + 1)
    # If some cell is empty, min = 0, else min over counts
    if len(counts) < total_cells:
        m = 0
    else:
        # every cell has at least one strawberry
        m = min(counts.values())

    print(m, M)

if __name__ == "__main__":
    main()