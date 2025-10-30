import sys
import bisect

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    # read sorted coordinates
    xs = [int(next(it)) for _ in range(n)]
    # read populations
    ps = [int(next(it)) for _ in range(n)]
    # build prefix sums of populations
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + ps[i]
    # number of queries
    q = int(next(it))
    out = []
    for _ in range(q):
        l = int(next(it)); r = int(next(it))
        # find leftmost index with xs[idx] >= l
        left = bisect.bisect_left(xs, l)
        # find rightmost index with xs[idx] <= r
        right = bisect.bisect_right(xs, r) - 1
        if left > right:
            out.append("0")
        else:
            # sum populations from left to right
            total = prefix[right+1] - prefix[left]
            out.append(str(total))
    sys.stdout.write("
".join(out))

# call main to execute
main()