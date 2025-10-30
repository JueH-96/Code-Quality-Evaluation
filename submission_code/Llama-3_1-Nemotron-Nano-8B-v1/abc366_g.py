def solve_bit(equations, n_vars):
    rows = []
    for mask, rhs in equations:
        rows.append((mask, rhs))
    
    rank = 0
    n = len(rows)
    for col in range(n_vars):
        pivot = -1
        for r in range(rank, n):
            if (rows[r][0] & (1 << col)) != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(n):
            if r != rank and (rows[r][0] & (1 << col)) != 0:
                rows[r] = (rows[r][0] ^ rows[rank][0], rows[r][1] ^ rows[rank][1])
        rank += 1
    
    solution = [0] * n_vars
    pivot_cols = set()
    for r in range(rank):
        mask, rhs = rows[r]
        pivot_col = -1
        for c in range(n_vars):
            if (mask & (1 << c)) != 0:
                pivot_col = c
                break
        if pivot_col == -1:
            continue
        pivot_cols.add(pivot_col)
        solution[pivot_col] = rhs
    
    free_vars = []
    for c in range(n_vars):
        if c not in pivot_cols:
            free_vars.append(c)
    if all(v == 0 for v in solution) and len(free_vars) > 0:
        solution[free_vars[0]] = 1
        solution = [0] * n_vars
        for r in range(rank):
            mask, rhs = rows[r]
            pivot_col = -1
            for c in range(n_vars):
                if (mask & (1 << c)) != 0:
                    pivot_col = c
                    break
            if pivot_col == -1:
                continue
            s = 0
            for c_var in range(n_vars):
                if c_var == pivot_col:
                    continue
                if (mask & (1 << c_var)) != 0:
                    s ^= solution[c_var]
            solution[pivot_col] = s ^ rhs
    
    return solution

n, m = map(int, input().split())
adj = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    u -= 1
    v -= 1
    adj[u].append(v)
    adj[v].append(u)

degrees = [len(adj[i]) for i in range(n)]

bits_solution = []
for bit in range(60):
    equations = []
    for v in range(n):
        if degrees[v] == 0:
            continue
        mask = 0
        for u in adj[v]:
            mask |= 1 << u
        equations.append((mask, 0))
    solution = solve_bit(equations, n)
    bits_solution.append(solution)

x = [0] * n
for i in range(n):
    val = 0
    for bit in range(60):
        val |= (bits_solution[bit][i] << bit)
    x[i] = val

for i in range(n):
    if degrees[i] == 0:
        if x[i] == 0:
            x[i] = 1

valid = all(xi != 0 for xi in x)

if valid:
    print("Yes")
    print(' '.join(map(str, x)))
else:
    print("No")