import sys
import bisect


def build_prefix(arr):
    """
    returns [0, v1, v1+v2, ...]
    """
    pref = [0]
    s = 0
    for v in arr:
        s += v
        pref.append(s)
    return pref


def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    pulls = []      # type 0
    regulars = []   # type 1
    openers = []    # type 2 (store their capacities)

    for _ in range(N):
        t = int(next(it))
        x = int(next(it))
        if t == 0:
            pulls.append(x)
        elif t == 1:
            regulars.append(x)
        else:          # t == 2
            openers.append(x)

    # sort descending, because we only ever need “top” values
    pulls.sort(reverse=True)
    regulars.sort(reverse=True)
    openers.sort(reverse=True)

    # prefix sums (prefix[0] == 0)
    pull_pref = build_prefix(pulls)
    reg_pref = build_prefix(regulars)
    cap_pref = build_prefix(openers)          # cumulative capacities supplied by the first k openers

    P = len(pulls)
    R = len(regulars)
    O = len(openers)

    # minimal number of openers that MUST be taken just to have M items in total
    min_openers_needed = max(0, M - (P + R))

    best = 0
    # we cannot open more than M - min_openers_needed regular cans (since we need room for openers)
    max_r = min(R, M - min_openers_needed)

    for r in range(max_r + 1):                       # r … number of opened regular cans
        # smallest k whose total capacity is at least r
        k_cap = bisect.bisect_left(cap_pref, r)
        if k_cap > O:                                # not enough capacity overall
            continue

        k = max(k_cap, min_openers_needed)           # real number of openers we take
        if k > O or k + r > M:                       # impossible combination
            continue

        remaining_slots = M - k - r                  # these slots will be filled with pull–tab cans or unopened regulars
        # pull-tabs give happiness, unopened regulars give 0
        pull_cnt = remaining_slots if remaining_slots <= P else P
        happiness = reg_pref[r] + pull_pref[pull_cnt]
        best = max(best, happiness)

    print(best)


if __name__ == "__main__":
    main()