import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    xs = [0]*n
    hs = [0]*n
    for i in range(n):
        xs[i] = int(next(it))
        hs[i] = int(next(it))

    max_h = -1.0          # maximum height at which some building is NOT visible
    found = False         # whether such a height exists

    for i in range(n - 1):
        x1, h1 = xs[i], hs[i]
        x2, h2 = xs[i + 1], hs[i + 1]

        # Check if top–slope of building i is not smaller than that of i+1 at height 0
        # i.e. H_i / X_i >= H_{i+1} / X_{i+1}
        if h1 * x2 >= h2 * x1:
            # critical height where their top slopes become equal
            numerator   = x2 * h1 - x1 * h2        # >= 0  by the test above
            denominator = x2 - x1                  # > 0   because X is strictly increasing
            h_crit = numerator / denominator       # float division is fine (≤ 1e9)
            if h_crit > max_h:
                max_h = h_crit
            found = True

    if not found:
        # All buildings already visible from height 0
        print("-1")
    else:
        # Print with plenty of decimal places (1e-9 accuracy required)
        print("{:.18f}".format(max_h))


if __name__ == "__main__":
    main()