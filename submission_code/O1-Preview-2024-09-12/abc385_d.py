# YOUR CODE HERE

import sys
import threading
import bisect

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N, M, S_x, S_y = map(int, sys.stdin.readline().split())

    houses = set()
    x_to_ys = {}
    y_to_xs = {}

    for _ in range(N):
        X_i, Y_i = map(int, sys.stdin.readline().split())
        houses.add((X_i, Y_i))
        if X_i not in x_to_ys:
            x_to_ys[X_i] = []
        x_to_ys[X_i].append(Y_i)
        if Y_i not in y_to_xs:
            y_to_xs[Y_i] = []
        y_to_xs[Y_i].append(X_i)

    # Sort the lists for bisect
    for x in x_to_ys:
        x_to_ys[x].sort()
    for y in y_to_xs:
        y_to_xs[y].sort()

    cur_x, cur_y = S_x, S_y
    visited_houses = set()

    for _ in range(M):
        D_i, C_i = sys.stdin.readline().split()
        C_i = int(C_i)
        if D_i == 'U':
            new_x, new_y = cur_x, cur_y + C_i
            # Movement along x = cur_x, y from cur_y to new_y
            x = cur_x
            y1, y2 = sorted([cur_y, new_y])
            if x in x_to_ys:
                ys = x_to_ys[x]
                idx_start = bisect.bisect_left(ys, y1 + 1e-6)
                idx_end = bisect.bisect_right(ys, y2 - 1e-6)
                for idx in range(idx_start, idx_end):
                    house_y = ys[idx]
                    visited_houses.add((x, house_y))
        elif D_i == 'D':
            new_x, new_y = cur_x, cur_y - C_i
            # Movement along x = cur_x, y from cur_y to new_y
            x = cur_x
            y1, y2 = sorted([cur_y, new_y])
            if x in x_to_ys:
                ys = x_to_ys[x]
                idx_start = bisect.bisect_left(ys, y1 + 1e-6)
                idx_end = bisect.bisect_right(ys, y2 - 1e-6)
                for idx in range(idx_start, idx_end):
                    house_y = ys[idx]
                    visited_houses.add((x, house_y))
        elif D_i == 'L':
            new_x, new_y = cur_x - C_i, cur_y
            # Movement along y = cur_y, x from cur_x to new_x
            y = cur_y
            x1, x2 = sorted([cur_x, new_x])
            if y in y_to_xs:
                xs = y_to_xs[y]
                idx_start = bisect.bisect_left(xs, x1 + 1e-6)
                idx_end = bisect.bisect_right(xs, x2 - 1e-6)
                for idx in range(idx_start, idx_end):
                    house_x = xs[idx]
                    visited_houses.add((house_x, y))
        elif D_i == 'R':
            new_x, new_y = cur_x + C_i, cur_y
            # Movement along y = cur_y, x from cur_x to new_x
            y = cur_y
            x1, x2 = sorted([cur_x, new_x])
            if y in y_to_xs:
                xs = y_to_xs[y]
                idx_start = bisect.bisect_left(xs, x1 + 1e-6)
                idx_end = bisect.bisect_right(xs, x2 - 1e-6)
                for idx in range(idx_start, idx_end):
                    house_x = xs[idx]
                    visited_houses.add((house_x, y))
        else:
            # Should not happen
            pass
        # Update Santa's position
        cur_x, cur_y = new_x, new_y

    # Output the result
    print(f"{cur_x} {cur_y} {len(visited_houses)}")


threading.Thread(target=main).start()