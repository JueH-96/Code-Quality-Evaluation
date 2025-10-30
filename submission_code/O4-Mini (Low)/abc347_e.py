import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    xs = [int(next(it)) for _ in range(Q)]
    # Track active status and start times
    active = [False] * (N + 1)
    start_time = [0] * (N + 1)
    # For each element, store list of (l, r) intervals (1-based, inclusive)
    intervals = [[] for _ in range(N + 1)]
    # We'll record k[1..Q]: size of S after each toggle
    k = [0] * (Q + 1)
    size = 0
    for i, x in enumerate(xs, start=1):
        if not active[x]:
            # insert
            active[x] = True
            start_time[x] = i
            size += 1
        else:
            # remove
            active[x] = False
            l = start_time[x]
            r = i - 1
            if l <= r:
                intervals[x].append((l, r))
            size -= 1
        k[i] = size

    # close intervals for those still active
    for x in range(1, N + 1):
        if active[x]:
            l = start_time[x]
            r = Q
            if l <= r:
                intervals[x].append((l, r))

    # build prefix sums of k
    P = [0] * (Q + 1)
    for i in range(1, Q + 1):
        P[i] = P[i - 1] + k[i]

    # compute answers
    A = [0] * (N + 1)
    for x in range(1, N + 1):
        total = 0
        for l, r in intervals[x]:
            total += P[r] - P[l - 1]
        A[x] = total

    # output
    out = " ".join(str(A[i]) for i in range(1, N + 1))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()