import sys
import bisect

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # 1-based prefix counters
    pre1 = [0] * (N + 1)
    pre2 = [0] * (N + 1)
    slash = []                     # positions of '/'

    for i, ch in enumerate(S, 1):
        pre1[i] = pre1[i - 1] + (ch == '1')
        pre2[i] = pre2[i - 1] + (ch == '2')
        if ch == '/':
            slash.append(i)

    total_slash = len(slash)
    out_lines = []

    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())

        # find indices of first slash >= L and last slash <= R
        left_idx = bisect.bisect_left(slash, L)
        right_idx = bisect.bisect_right(slash, R) - 1
        if left_idx > right_idx:            # no slash in the substring
            out_lines.append('0')
            continue

        # helper to compute g(p) and left/right counts in O(1)
        def counts(pos):
            # pos is an index into the whole string (1-based)
            left  = pre1[pos - 1] - pre1[L - 1]
            right = pre2[R] - pre2[pos]
            return left, right, left - right

        # monotone g -> binary search first slash with g >= 0
        lo = left_idx
        hi = right_idx
        left, right, diff = counts(slash[lo])
        if diff >= 0:                       # already non-negative
            k_hi = right                    # min(left,right) == right
            ans = 2 * k_hi + 1
            out_lines.append(str(ans))
            continue

        left, right, diff = counts(slash[hi])
        if diff < 0:                        # never crosses zero
            k_lo = left                     # min(left,right) == left
            ans = 2 * k_lo + 1
            out_lines.append(str(ans))
            continue

        # proper crossing: g(lo) < 0 , g(hi) >= 0
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            _, _, d = counts(slash[mid])
            if d < 0:
                lo = mid
            else:
                hi = mid

        # now hi is first with g >= 0, lo last with g < 0
        k_lo = counts(slash[lo])[0]     # = left_1(lo)
        k_hi = counts(slash[hi])[1]     # = right_2(hi)
        k = k_lo if k_lo > k_hi else k_hi
        out_lines.append(str(2 * k + 1))

    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()