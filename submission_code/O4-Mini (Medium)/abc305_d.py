import sys
import threading

def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    # Build prefix sums of sleep up to each A[i]
    # Intervals are [A[1],A[2]], [A[3],A[4]], ... in 0-based A[1],A[2], i.e. start at odd, end at even.
    S = [0] * n
    for i in range(1, n):
        if i % 2 == 0:
            # A[i] is wake time, end of sleep from A[i-1]
            S[i] = S[i-1] + (A[i] - A[i-1])
        else:
            # A[i] is sleep start, no addition
            S[i] = S[i-1]

    def f(t):
        # total sleep in [0, t)
        j = bisect_left(A, t)
        # if inside a sleep interval (j even >0), t is before A[j] == end of that interval
        if j > 0 and j % 2 == 0:
            # sum before this interval plus partial [A[j-1], t)
            return S[j-1] + (t - A[j-1])
        else:
            # not inside a sleep interval, we have full sums up to A[j]
            # j is <= n-1 because t <= A[n-1]
            return S[j]

    q = int(next(it))
    out = []
    for _ in range(q):
        l = int(next(it)); r = int(next(it))
        # answer is f(r) - f(l)
        ans = f(r) - f(l)
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()