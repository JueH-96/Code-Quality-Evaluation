import sys
import threading

def main():
    import sys
    import bisect

    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    Q = int(input().strip())

    # Build sleep intervals: for i=1,3,5,... (0-based) we have interval [A[i], A[i+1])
    K = (N - 1) // 2
    starts = [0] * K
    ends   = [0] * K
    for i in range(K):
        starts[i] = A[2*i + 1]
        ends[i]   = A[2*i + 2]

    # Prefix sums of full sleep durations
    # PS[i] = total sleep in intervals [0..i-1]
    PS = [0] * (K+1)
    for i in range(K):
        PS[i+1] = PS[i] + (ends[i] - starts[i])

    out = []
    for _ in range(Q):
        l, r = map(int, input().split())
        # find first interval with end > l
        j = bisect.bisect_right(ends, l)
        # find last interval with start < r
        k = bisect.bisect_left(starts, r) - 1

        if j > k or j >= K or k < 0:
            # no overlapping intervals
            out.append("0")
            continue

        # full sum over intervals j..k
        total = PS[k+1] - PS[j]

        # subtract left overflow if l cuts into interval j
        if l > starts[j]:
            # we included (ends[j]-starts[j]) fully, but only want (ends[j]-l)
            total -= (l - starts[j])

        # subtract right overflow if r cuts into interval k
        if r < ends[k]:
            # we included (ends[k]-starts[k]) fully, but only want (r-starts[k])
            total -= (ends[k] - r)

        out.append(str(total))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()