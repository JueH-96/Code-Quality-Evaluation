# YOUR CODE HERE
def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    # n buildings with coordinate and height
    X = [0.0] * n
    H = [0.0] * n
    for i in range(n):
        X[i] = float(next(it))
        H[i] = float(next(it))
    # For one building, no blocking occurs so answer = -1.
    if n == 1:
        sys.stdout.write("-1")
        return

    # Precompute r[i] = H[i]/X[i]
    r = [0.0] * n
    for i in range(n):
        r[i] = H[i] / X[i]

    # For a given observer height h (at (0, h)) we want to decide whether every building is visible.
    # One may show that building 0 is always visible and that for i >= 1 the condition
    #    r[i] > (h / X[i]) + max_{j < i} (r[j] - h/X[j])
    # is necessary and sufficient for building i to be seen.
    # (Note that when h is 0, the condition becomes r[i] > max_{j < i} r[j].)
    def all_visible(h):
        p = r[0] - h / X[0]  # for building 0 no blocking; set our prefix value
        for i in range(1, n):
            cur = r[i] - h / X[i]
            if cur <= p:  # if equality (or less) then building i cannot be seen
                return False
            if cur > p:
                p = cur
        return True

    # If from ground level (h = 0) all buildings are visible then answer = -1.
    if all_visible(0.0):
        sys.stdout.write("-1")
        return

    # Otherwise, there is a unique threshold h₀ so that for h <= h₀ not all building are visible
    # and for h > h₀ every building is visible.
    # We now binary–search for the supremum of h where not-all-visible holds.
    lo = 0.0
    hi = 1.0
    # Increase hi (exponential search) until all_visible(hi) becomes True.
    while not all_visible(hi):
        hi *= 2.0
        if hi > 1e20:
            break

    # Binary search for the transition value. (We want the maximum h with all_visible(h)==False.)
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if all_visible(mid):
            hi = mid
        else:
            lo = mid
    # lo is the maximum height from which not all buildings are visible.
    sys.stdout.write("{:.18f}".format(lo))
    
if __name__ == '__main__':
    main()