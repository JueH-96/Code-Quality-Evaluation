import sys

def v2(x: int) -> int:
    """return exponent of 2 in x"""
    return (x & -x).bit_length() - 1  # faster than loop, x>0


def solve_case(n: int, k: int, arr) -> int:
    if k == 2:
        # need at least one even number
        for x in arr:
            if x & 1 == 0:
                return 0
        return 1  # every increment of an odd number makes it even
    if k == 3:
        best = 3  # larger than possible
        for x in arr:
            r = x % 3
            if r == 0:
                return 0
            best = min(best, 3 - r)
        return best
    if k == 5:
        best = 5
        for x in arr:
            r = x % 5
            if r == 0:
                return 0
            best = min(best, 5 - r)
        return best
    # k == 4
    total_pow2 = 0
    min_cost_div4 = 4  # minimal extra needed to reach a multiple of 4
    odd_cnt = 0
    for x in arr:
        total_pow2 += v2(x)
        rem4 = x & 3  # x % 4
        if rem4 == 0:
            min_cost_div4 = 0
        else:
            min_cost_div4 = min(min_cost_div4, 4 - rem4)
        if x & 1:  # odd
            odd_cnt += 1

    if total_pow2 >= 2:
        return 0
    if total_pow2 == 1:
        # need only one extra power of two
        return min(1, min_cost_div4)  # 1 comes from making any odd number even
    # total_pow2 == 0   (all numbers are odd)
    return min(min_cost_div4, 2)      # 2 ops = make any two odds even


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    t = int(next(it))
    out_lines = []
    for _ in range(t):
        n = int(next(it))
        k = int(next(it))
        arr = [int(next(it)) for _ in range(n)]
        out_lines.append(str(solve_case(n, k, arr)))
    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()