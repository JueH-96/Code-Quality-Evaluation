import sys
import threading

def main():
    import sys
    data = sys.stdin.readline().split()
    if not data:
        data = sys.stdin.readline().split()
    N, Q = map(int, data)
    xs = list(map(int, sys.stdin.readline().split()))
    # s[i]: size of S after i-th toggle (1-indexed)
    s = [0] * (Q + 1)
    present = [False] * (N + 1)
    # cur_entry[x] = time i when x was most recently inserted (and not yet removed)
    cur_entry = [0] * (N + 1)
    # list of intervals (x, l, r): x is active on [l, r)
    intervals = []
    c = 0  # current size of S
    for i in range(1, Q + 1):
        x = xs[i - 1]
        if not present[x]:
            # insert
            present[x] = True
            c += 1
            s[i] = c
            cur_entry[x] = i
        else:
            # remove
            present[x] = False
            c -= 1
            s[i] = c
            l = cur_entry[x]
            # interval [l, i)
            intervals.append((x, l, i))
            cur_entry[x] = 0

    # any still present at end: interval [l, Q+1)
    end_time = Q + 1
    for x in range(1, N + 1):
        if present[x]:
            l = cur_entry[x]
            intervals.append((x, l, end_time))

    # build prefix sums of s
    ps = [0] * (Q + 1)
    for i in range(1, Q + 1):
        ps[i] = ps[i - 1] + s[i]

    # compute final A
    A = [0] * (N + 1)
    for x, l, r in intervals:
        # sum s[l..r-1] = ps[r-1] - ps[l-1]
        A[x] += ps[r - 1] - ps[l - 1]

    # output
    out = sys.stdout
    res = " ".join(str(A[i]) for i in range(1, N + 1))
    out.write(res)

if __name__ == "__main__":
    main()