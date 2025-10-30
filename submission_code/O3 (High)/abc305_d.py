import sys
import bisect


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    idx = 0

    # number of recorded time points (odd)
    N = data[idx]
    idx += 1

    # time points themselves
    A = data[idx: idx + N]
    idx += N

    # number of queries
    Q = data[idx]
    idx += 1

    # ------------------------------------------------------------
    # build prefix array:
    # pref[i] = total sleeping minutes in the half-open interval [0, A[i])
    # ------------------------------------------------------------
    pref = [0] * N
    for i in range(1, N):
        pref[i] = pref[i - 1]
        if (i - 1) & 1:                       # in [A[i-1], A[i}) Takahashi was asleep
            pref[i] += A[i] - A[i - 1]

    # helper: total minutes asleep in [0, x)
    def slept_until(x: int) -> int:
        pos = bisect.bisect_right(A, x)       # first index with A[pos] > x
        j = pos - 1                           # A[j] ≤ x  < A[j+1]  (or j = N-1)
        res = pref[j]
        if j & 1:                             # still in a sleeping segment
            res += x - A[j]
        return res

    # process queries
    out_lines = []
    for _ in range(Q):
        l = data[idx]
        r = data[idx + 1]
        idx += 2
        out_lines.append(str(slept_until(r) - slept_until(l)))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()