import sys

def solve_bit(equations, N):
    rows = [list(row) for row in equations]
    rank = 0
    for col in range(N):
        # Find pivot in this column, from rank onwards
        pivot = -1
        for r in range(rank, len(rows)):
            if rows[r][col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap with the rank row
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        # Eliminate this column in other rows
        for r in range(len(rows)):
            if r != rank and rows[r][col] == 1:
                for j in range(col, N):
                    rows[r][j] ^= rows[rank][j]
        rank += 1

    # Find pivot columns
    pivot_cols = set()
    for r in range(rank):
        # find first column with 1 in this row
        for c in range(N):
            if rows[r][c] == 1:
                pivot_cols.add(c)
                break

    # Generate solution
    solution = [0]*N
    # Assign 1 to free variables
    for c in range(N):
        if c not in pivot_cols:
            solution[c] = 1

    # Back substitute
    for r in reversed(range(rank)):
        # find pivot column
        pivot_col = -1
        for c in range(N):
            if rows[r][c] == 1:
                pivot_col = c
                break
        # calculate sum of rows[r][c] * solution[c] for c > pivot_col
        s = 0
        for c in range(pivot_col + 1, N):
            s ^= (rows[r][c] & solution[c])
        solution[pivot_col] = s

    return solution

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N+1)]  # 1-based
    degrees = [0]*(N+1)
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    # Prepare solutions for each bit
    solutions = [ [0]*N for _ in range(60) ]
    for bit in range(60):
        equations = []
        for v in range(1, N+1):
            if degrees[v] >= 1:
                row = [0]*N
                for neighbor in adj[v]:
                    row[neighbor-1] = 1  # neighbor is 1-based, converted to 0-based
                equations.append(row)
        # Solve for this bit
        sol = solve_bit(equations, N)
        for i in range(N):
            solutions[bit][i] = sol[i]

    # Compute X for each vertex
    X = [0]*(N+1)  # 1-based
    for v in range(1, N+1):
        x = 0
        for bit in range(60):
            if solutions[bit][v-1]:
                x += (1 << bit)
        X[v] = x

    # Check if any vertex with degree >=1 has X[v] == 0
    has_invalid = False
    for v in range(1, N+1):
        if degrees[v] >= 1 and X[v] == 0:
            has_invalid = True
            break

    if has_invalid:
        print("No")
    else:
        print("Yes")
        print(' '.join(map(str, X[1:N+1])))

if __name__ == '__main__':
    main()