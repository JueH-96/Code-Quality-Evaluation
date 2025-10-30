import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1

    pull_tabs = []
    regular_cans = []
    openers = []

    for _ in range(N):
        T = int(input[idx]); idx += 1
        X = int(input[idx]); idx += 1
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular_cans.append(X)
        else:
            openers.append(X)

    # Sort each list in descending order
    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    openers.sort(reverse=True)

    # Compute prefix sums
    pull_tabs_sum = [0]
    for x in pull_tabs:
        pull_tabs_sum.append(pull_tabs_sum[-1] + x)

    regular_sum = [0]
    for x in regular_cans:
        regular_sum.append(regular_sum[-1] + x)

    opener_cap_sum = [0]
    for x in openers:
        opener_cap_sum.append(opener_cap_sum[-1] + x)

    O = len(openers)
    R = len(regular_cans)
    max_total = 0

    a_max = min(M, len(pull_tabs))
    for a in range(0, a_max + 1):
        rem = M - a
        if rem < 0:
            continue

        low = 0
        high = min(rem, R)
        best_b = 0

        while low <= high:
            mid = (low + high) // 2
            # Find the minimal c where opener_cap_sum[c] >= mid
            c_min = bisect.bisect_left(opener_cap_sum, mid)
            if c_min > O:
                # No such c exists
                high = mid - 1
            else:
                if c_min <= (rem - mid):
                    best_b = mid
                    low = mid + 1
                else:
                    high = mid - 1

        current_total = pull_tabs_sum[a] + (regular_sum[best_b] if best_b <= R else 0)
        if current_total > max_total:
            max_total = current_total

    print(max_total)

if __name__ == "__main__":
    main()