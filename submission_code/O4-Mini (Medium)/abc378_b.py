def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    qs = [0]*N
    rs = [0]*N
    for i in range(N):
        q = int(next(it)); r = int(next(it))
        qs[i] = q; rs[i] = r
    Q = int(next(it))
    out = []
    for _ in range(Q):
        t = int(next(it)) - 1
        d = int(next(it))
        q = qs[t]; r = rs[t]
        # delta = smallest non-negative x such that (d + x) % q == r
        delta = (r - (d % q) + q) % q
        out.append(str(d + delta))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()