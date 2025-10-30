def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        adj[u].append(v)
        adj[v].append(u)

    equations = []
    for v in range(1, N + 1):
        if len(adj[v]) == 0:
            continue
        eq = [0] * (N + 1)
        for u in adj[v]:
            eq[u] = 1
        equations.append(eq)

    row = 0
    pivot_cols = {}
    for col in range(1, N + 1):
        pivot_row = None
        for r in range(row, len(equations)):
            if equations[r][col] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        equations[row], equations[pivot_row] = equations[pivot_row], equations[row]

        for r in range(len(equations)):
            if r != row and equations[r][col] == 1:
                for c in range(1, N + 1):
                    equations[r][c] ^= equations[row][c]

        pivot_cols[col] = row
        row += 1

    forced_zero = set()
    for r in range(len(equations)):
        pivot = None
        for c in range(1, N + 1):
            if equations[r][c] == 1:
                pivot = c
                break
        if pivot is None:
            continue
        has_other = False
        for c in range(1, N + 1):
            if c != pivot and equations[r][c] == 1:
                has_other = True
                break
        if not has_other:
            forced_zero.add(pivot)

    if forced_zero:
        print("No")
        return

    pivot_vars = set(pivot_cols.keys())
    free_vars = [c for c in range(1, N + 1) if c not in pivot_vars]

    X = [0] * (N + 1)
    for i, var in enumerate(free_vars):
        X[var] = 1 << i

    for pivot_col in pivot_cols:
        r = pivot_cols[pivot_col]
        s = 0
        for c in range(1, N + 1):
            if c == pivot_col:
                continue
            if equations[r][c]:
                s ^= X[c]
        X[pivot_col] = s

    for v in range(1, N + 1):
        if X[v] == 0:
            print("No")
            return

    print("Yes")
    print(' '.join(map(str, X[1:])))

if __name__ == '__main__':
    main()