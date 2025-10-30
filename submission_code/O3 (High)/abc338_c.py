import sys

# ---------- auxiliary integer helpers ---------- #
def ceil_div(a: int, b: int) -> int:
    """
    Return ceil(a / b) for integers a, b (b != 0),
    working for every sign combination.
    """
    return -((-a) // b)


def feasible(total: int, Q, A, B) -> bool:
    """
    Determine whether it is possible to cook `total` servings in
    total (x of dish A and total-x of dish B) without exceeding
    the stock `Q`.  All arrays have the same length N (N <= 10).
    """
    lower = 0          # lower bound for x
    upper = total      # upper bound for x

    for q, a, b in zip(Q, A, B):
        diff = a - b   # coefficient of x in the inequality

        if diff == 0:                          # a == b
            if a * total > q:                  # need a*total grams
                return False                   # impossible
            # otherwise this ingredient does not restrict x
            continue

        if diff > 0:                           # (a - b) > 0  ->  upper bound
            max_x = (q - b * total) // diff    # biggest x allowed
            if max_x < 0:                      # even x = 0 is impossible
                return False
            upper = min(upper, max_x)
        else:                                  # diff < 0  ->  lower bound
            min_x = ceil_div(q - b * total, diff)  # smallest x allowed
            lower = max(lower, min_x)

        if lower > upper:                      # bounds crossed -> impossible
            return False

    # final feasibility check within [0, total]
    return 0 <= lower <= upper


# ---------- main routine ---------- #
def main() -> None:
    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:                              # empty input guard
        return
    idx = 0
    N = data[idx]; idx += 1

    Q = data[idx: idx + N]; idx += N
    A = data[idx: idx + N]; idx += N
    B = data[idx: idx + N]

    # a quick upper bound for the total number of servings:
    # each serving consumes at least 1 gram of some ingredient,
    # therefore total servings cannot exceed sum(Q_i)
    total_stock = sum(Q)
    min_positive = min(x for row in (A, B) for x in row if x > 0)
    high = total_stock // min_positive + 1     # safe inclusive upper bound

    # binary search on the answer
    low, ans = 0, 0
    while low <= high:
        mid = (low + high) // 2
        if feasible(mid, Q, A, B):
            ans = mid
            low = mid + 1    # try larger totals
        else:
            high = mid - 1   # reduce total

    print(ans)


if __name__ == "__main__":
    main()