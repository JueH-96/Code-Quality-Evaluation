def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    t = int(data[0])
    idx = 1
    out_lines = []

    for _ in range(t):
        n = int(data[idx])
        k = int(data[idx + 1])
        idx += 2
        s = data[idx]
        idx += 1

        # Collect positions of 'B' (black cells)
        blacks = [i for i, c in enumerate(s) if c == 'B']

        if not blacks:
            out_lines.append("0")
            continue

        ops = 0  # count of operations
        i = 0    # pointer in blacks list
        m = len(blacks)
        while i < m:
            # p is the leftmost uncovered black cell index.
            p = blacks[i]
            # To cover cell p, we can choose any consecutive block of k cells starting in the range
            # [max(0, p - k + 1), min(p, n - k)]. To cover as far right as possible,
            # choose the maximum starting index.
            start_index = p if p <= n - k else n - k
            # The operation covers from start_index to start_index + k - 1.
            R = start_index + k - 1
            ops += 1
            # Skip all black cell positions that are covered by the current operation.
            while i < m and blacks[i] <= R:
                i += 1
        out_lines.append(str(ops))

    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()