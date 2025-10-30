import sys
import bisect
from array import array

# ---------- helper -----------------------------------------------------------
def build_cost_list(points, D):
    """
    For one coordinate axis (x or y) build and return an array('q') that contains
    S(t) = sum_i |t - p_i| for every integer t whose value is not larger than D.
    """
    points.sort()
    n = len(points)

    # prefix sums
    pref = [0] * (n + 1)
    for i, v in enumerate(points):
        pref[i + 1] = pref[i] + v
    total = pref[n]

    lo = points[0] - D                # the farthest possible place that may satisfy S<=D
    hi = points[-1] + D               # (same for the right side)

    # initial position
    x = lo
    idx = bisect.bisect_right(points, x)   # number of points <= x
    left_cnt  = idx
    right_cnt = n - idx
    left_sum  = pref[left_cnt]
    right_sum = total - left_sum
    cost = left_cnt * x - left_sum + right_sum - right_cnt * x  # S(lo)

    res = array('q')
    while True:
        if cost <= D:
            res.append(cost)

        if x == hi:                   # finished the sweep
            break

        # move to the next integer coordinate
        slope = left_cnt - right_cnt          # S(x+1) - S(x)
        cost  = cost + slope
        nx    = x + 1

        # points that are exactly at nx now move to the left side
        while idx < n and points[idx] == nx:
            idx += 1
            left_cnt  += 1
            right_cnt -= 1

        x = nx
    return res
# -----------------------------------------------------------------------------


def main() -> None:
    read = sys.stdin.buffer.readline
    while True:
        first = read()
        if first.strip():
            break
    N, D = map(int, first.split())

    xs, ys = [], []
    for _ in range(N):
        x, y = map(int, read().split())
        xs.append(x)
        ys.append(y)

    # build the cost lists for x–coordinate and y–coordinate separately
    cost_x = build_cost_list(xs, D)
    cost_y = build_cost_list(ys, D)

    if not cost_x or not cost_y:          # no feasible x or y at all
        print(0)
        return

    cost_x.sort()
    cost_y.sort()

    # two–pointer to count pairs with sum <= D
    ans = 0
    j = len(cost_y) - 1
    for cx in cost_x:
        limit = D - cx
        while j >= 0 and cost_y[j] > limit:
            j -= 1
        if j < 0:
            break
        ans += (j + 1)

    print(ans)


if __name__ == "__main__":
    main()