import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    R = [int(next(it)) for _ in range(n)]
    R.sort()
    # build prefix sums
    prefix = [0]*n
    s = 0
    for i, r in enumerate(R):
        s += r
        prefix[i] = s
    # process queries
    out = []
    from bisect import bisect_right
    for _ in range(q):
        X = int(next(it))
        # number of sleighs is the largest k with prefix[k-1] <= X
        # bisect_right gives index of first prefix > X
        cnt = bisect_right(prefix, X)
        out.append(str(cnt))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()