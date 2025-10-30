# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
    except StopIteration:
        return

    # Build per–row constraints.
    # For row i, default: L = 0 and H = N+1, meaning f(i) (the number of black cells) can be any integer in [0,N].
    row_data = {}
    for _ in range(M):
        try:
            r = int(next(it))
            y = int(next(it))
            col = next(it)
        except StopIteration:
            break
        if r not in row_data:
            row_data[r] = [0, N+1]  # [L, H]
        if col == 'B':
            # black cell: require that f(r) >= y.
            if y > row_data[r][0]:
                row_data[r][0] = y
        else:
            # white cell: require that f(r) < y, i.e. f(r) <= y-1.
            if y < row_data[r][1]:
                row_data[r][1] = y

    # For every row that has a constraint, the allowed values for f(i) are in [L, H-1].
    constrained = []
    for r, (L, H) in row_data.items():
        if L >= H:
            sys.stdout.write("No")
            return
        constrained.append((r, L, H - 1))
    constrained.sort(key=lambda x: x[0])  # sort the rows in increasing order (top to bottom)

    # We need to choose a non-increasing sequence f(1) >= f(2) >= ... >= f(N)
    # For unconstrained rows (those not in constrained) the allowed interval is [0, N] so no worry.
    # We will greedily “simulate” the assignment on the constrained rows.
    cur = N  # initially, we may assign up to N for the topmost row.
    for (r, L, U) in constrained:
        if cur < L:
            sys.stdout.write("No")
            return
        cur = min(cur, U)
    sys.stdout.write("Yes")

if __name__ == '__main__':
    main()