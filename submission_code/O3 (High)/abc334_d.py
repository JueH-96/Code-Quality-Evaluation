import sys
import bisect

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n, q = data[0], data[1]
    r = data[2:2 + n]
    queries = data[2 + n:]

    # sort the requirements so that the cheapest sleighs are taken first
    r.sort()
    # prefix sums: prefix[i] = total reindeer needed for first (i+1) cheapest sleighs
    prefix = [0] * n
    s = 0
    for i, val in enumerate(r):
        s += val
        prefix[i] = s

    out_lines = []
    for x in queries:
        # largest k such that prefix[k-1] <= x
        cnt = bisect.bisect_right(prefix, x)
        out_lines.append(str(cnt))

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()