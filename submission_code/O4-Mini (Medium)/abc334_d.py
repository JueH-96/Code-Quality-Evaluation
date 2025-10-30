import sys
import threading

def main():
    import sys
    from bisect import bisect_right

    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    R = [int(next(it)) for _ in range(N)]
    R.sort()
    # Build prefix sums: pref[0] = 0, pref[i] = sum of first i R's
    pref = [0] * (N + 1)
    for i in range(N):
        pref[i+1] = pref[i] + R[i]
    out = []
    for _ in range(Q):
        X = int(next(it))
        # Find largest k such that pref[k] <= X
        # bisect_right gives first index where pref[idx] > X
        # so answer is idx-1
        idx = bisect_right(pref, X)
        out.append(str(idx - 1))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()