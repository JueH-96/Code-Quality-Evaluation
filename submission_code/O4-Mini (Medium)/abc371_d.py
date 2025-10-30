import sys
import threading

def main():
    import sys
    from bisect import bisect_left, bisect_right

    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    xs = [int(next(it)) for _ in range(n)]
    ps = [int(next(it)) for _ in range(n)]
    # build prefix sums
    pre = [0] * (n + 1)
    for i in range(n):
        pre[i+1] = pre[i] + ps[i]

    q = int(next(it))
    out = []
    for _ in range(q):
        L = int(next(it))
        R = int(next(it))
        # leftmost index with xs[idx] >= L
        l = bisect_left(xs, L)
        # rightmost index with xs[idx] <= R is bisect_right(xs, R) - 1
        r = bisect_right(xs, R) - 1
        if l <= r:
            ans = pre[r+1] - pre[l]
        else:
            ans = 0
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()