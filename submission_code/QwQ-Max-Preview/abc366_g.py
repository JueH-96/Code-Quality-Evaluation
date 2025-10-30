def solve_system(matrix, rhs):
    n_vars = len(matrix[0]) if matrix else 0
    augmented = [row + [rhs[i]] for i, row in enumerate(matrix)]
    rows = len(augmented)
    if rows == 0:
        return (True, [0]*n_vars)
    n_cols = len(augmented[0]) - 1
    rank = 0
    # Forward elimination
    for col in range(n_cols):
        # Find pivot row
        pivot = None
        for r in range(rank, rows):
            if augmented[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap current row with pivot row
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        # Eliminate all other rows
        for r in range(rows):
            if r != rank and augmented[r][col] == 1:
                for c in range(col, n_cols + 1):
                    augmented[r][c] ^= augmented[rank][c]
        rank += 1
    # Check for inconsistent equations
    for r in range(rows):
        if all(x == 0 for x in augmented[r][:n_cols]) and augmented[r][-1] == 1:
            return (False, None)
    # Back substitution
    solution = [0] * n_vars
    for r in range(rank):
        # Find leading variable in this row
        leading = -1
        for c in range(n_cols):
            if augmented[r][c] == 1:
                leading = c
                break
        if leading == -1:
            continue
        sum_val = 0
        for c in range(leading + 1, n_cols):
            sum_val ^= (augmented[r][c] & solution[c])
        solution[leading] = augmented[r][-1] ^ sum_val
    return (True, solution)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx +=1
    adj = [[] for _ in range(N+1)]  # 1-based
    for _ in range(M):
        u = int(input[idx])
        idx +=1
        v = int(input[idx])
        idx +=1
        adj[u].append(v)
        adj[v].append(u)
    # Build original system
    matrix = []
    rhs = []
    for v in range(1, N+1):
        if len(adj[v]) == 0:
            continue
        row = [0] * N
        for u in adj[v]:
            row[u-1] = 1  # variables are 0-based (x_0 is x_1)
        matrix.append(row)
        rhs.append(0)
    # Check original system
    solvable, _ = solve_system(matrix, rhs)
    if not solvable:
        print("No")
        return
    # For each vertex v, check if there's a solution with x_v=1
    solutions = []
    possible = True
    for v in range(1, N+1):
        # Create augmented system: original + x_v=1
        aug_matrix = [row.copy() for row in matrix]
        aug_row = [0] * N
        aug_row[v-1] = 1
        aug_matrix.append(aug_row)
        aug_rhs = rhs.copy() + [1]
        # Solve
        ok, solution = solve_system(aug_matrix, aug_rhs)
        if not ok:
            possible = False
            break
        solutions.append(solution)
    if not possible:
        print("No")
        return
    # Assign each vertex to a unique bit (v-1)
    X = [0] * N
    for k in range(60):
        if k < N:
            v = k + 1
            solution = solutions[v-1]
        else:
            solution = [0] * N
        for u in range(N):
            if solution[u] == 1:
                X[u] |= (1 << k)
    # Check if any X[u] is zero (shouldn't happen due to earlier checks)
    for u in range(N):
        if X[u] == 0:
            print("No")
            return
    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()