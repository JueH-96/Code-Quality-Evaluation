#!/usr/bin/env python3
import sys
def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # For each row x that appears in constraints, we keep [L, U]:
    #   L = minimum required r[x] by black cells (max y of B)
    #   U = maximum allowed r[x] by white cells (min y-1 of W)
    row_bounds = {}
    for _ in range(M):
        line = input().split()
        if not line:
            line = input().split()
        x = int(line[0])
        y = int(line[1])
        c = line[2]
        if x not in row_bounds:
            # initially row x can be anything in [0..N]
            row_bounds[x] = [0, N]
        L, U = row_bounds[x]
        if c == 'B':
            # black at (x,y) forces r[x] >= y
            if y > L:
                L = y
        else:
            # white at (x,y) forces r[x] < y  => r[x] <= y-1
            # so U = min(U, y-1)
            m = y - 1
            if m < U:
                U = m
        row_bounds[x] = [L, U]
    # Collect and sort the constrained rows by their index
    items = sorted(row_bounds.items(), key=lambda kv: kv[0])
    # We'll build a non-increasing sequence r[ ] by greedily assigning
    # r at the first constrained row = its U, then each next r = min(prev_r, U)
    # We must also ensure each r is >= its L, and prev_r >= its L to keep non-increasing.
    cur_max = N
    for x, (L, U) in items:
        # empty interval?
        if L > U:
            print("No")
            return
        # to keep r non-increasing, we need cur_max >= r[x] >= L
        # we can only pick r[x] up to min(cur_max, U), so that must be >= L
        if cur_max < L:
            print("No")
            return
        # update the maximum for the next row
        # we choose r[x] = min(cur_max, U)
        cur_max = min(cur_max, U)
    # all checks passed
    print("Yes")

if __name__ == "__main__":
    main()