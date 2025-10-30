def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    qs = [0] * (n + 1)
    rs = [0] * (n + 1)
    for i in range(1, n + 1):
        q_i = int(next(it))
        r_i = int(next(it))
        qs[i] = q_i
        rs[i] = r_i
    q_count = int(next(it))
    out_lines = []
    for _ in range(q_count):
        t = int(next(it))
        d = int(next(it))
        q_i = qs[t]
        r_i = rs[t]
        # compute minimal k >= 0 such that (d + k) % q_i == r_i
        rem = d % q_i
        # difference to reach r_i
        diff = (r_i - rem) % q_i
        out_lines.append(str(d + diff))
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()