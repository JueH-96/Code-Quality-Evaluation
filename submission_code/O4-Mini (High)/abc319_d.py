def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    L = list(map(int, input().split()))

    # Lower bound is the widest single word,
    # upper bound is putting all words on one line.
    lo = max(L)
    hi = sum(L) + (N - 1)

    # Binary search for minimum width W such that
    # the sentence can be displayed in <= M lines.
    while lo < hi:
        mid = (lo + hi) // 2
        lines = 1
        cur = L[0]
        for i in range(1, N):
            w = L[i]
            # Try to place in current line
            if cur + 1 + w <= mid:
                cur += 1 + w
            else:
                # Move to next line
                lines += 1
                cur = w
                if lines > M:
                    break
        # If we used at most M lines, mid is feasible
        if lines <= M:
            hi = mid
        else:
            lo = mid + 1

    print(lo)

main()