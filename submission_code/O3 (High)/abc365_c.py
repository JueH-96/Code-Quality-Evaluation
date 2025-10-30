import sys, bisect

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n, m = data[0], data[1]
    a = data[2:]
    
    total = sum(a)
    if total <= m:                 # Full reimbursement is affordable
        print("infinite")
        return

    a.sort()
    prefix = [0]
    for v in a:                    # prefix[i] = sum of first i elements (0-based)
        prefix.append(prefix[-1] + v)

    def cost(x: int) -> int:
        """Total subsidy when the upper limit is x."""
        idx = bisect.bisect_right(a, x)        # first position with a[idx] > x
        return prefix[idx] + (n - idx) * x

    low, high = 0, max(a)                      # high > 0 because a_i ≥ 1
    while low < high:
        mid = (low + high + 1) // 2            # upper mid
        if cost(mid) <= m:
            low = mid
        else:
            high = mid - 1

    print(low)

if __name__ == "__main__":
    main()