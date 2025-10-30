def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    edges = [[] for _ in range(N+1)]  # 1-based indexing
    degrees = [0] * (N + 1)

    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        edges[u].append(v)
        edges[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    # Check if all degrees are even
    all_even = True
    for i in range(1, N + 1):
        if degrees[i] % 2 != 0:
            all_even = False
            break
    if all_even:
        print("Yes")
        print(' '.join(['1'] * N))
        return

    # If M is 0, all degrees are zero, which is even
    if M == 0:
        print("Yes")
        print(' '.join(['1'] * N))
        return

    # Function to perform Gaussian elimination over GF(2)
    def gauss(matrix, n_vars):
        mat = [row for row in matrix]
        n = len(mat)
        rank = 0
        for col in reversed(range(n_vars)):  # Process columns from highest to lowest
            # Find the pivot row
            pivot = -1
            for r in range(rank, n):
                if (mat[r] >> (n_vars - 1 - col)) & 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap with the current rank row
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Eliminate this column in all other rows
            for r in range(n):
                if r != rank and ((mat[r] >> (n_vars - 1 - col)) & 1):
                    mat[r] ^= mat[rank]
            rank += 1
        # Assign variables
        solution = [0] * n_vars
        for r in range(rank):
            # Find the pivot column
            pivot_col = -1
            for c in reversed(range(n_vars)):
                if ((mat[r] >> (n_vars - 1 - c)) & 1):
                    pivot_col = c
                    break
            if pivot_col == -1:
                continue
            # Calculate the value for the pivot variable
            val = 0
            for c in range(pivot_col + 1, n_vars):
                if ((mat[r] >> (n_vars - 1 - c)) & 1):
                    val ^= solution[c]
            solution[pivot_col] = val
        return solution

    solutions = []
    for k in range(60):
        system = []
        for v in range(1, N + 1):
            if degrees[v] == 0:
                continue
            equation = 0
            for u in edges[v]:
                # Convert u to 0-based index
                equation |= (1 << (u - 1))
            system.append(equation)
        # Perform Gaussian elimination
        if not system:
            # No equations for this bit, assign all variables to 0 (but we'll check later)
            solutions.append([0] * N)
            continue
        try:
            sol = gauss(system, N)
        except:
            print("No")
            return
        solutions.append(sol)

    # Construct the variables
    x = [0] * N
    for k in range(60):
        sol = solutions[k]
        for v in range(N):
            if sol[v]:
                x[v] |= (1 << k)

    # Check if all variables are non-zero
    for v in range(N):
        if x[v] == 0:
            print("No")
            return

    print("Yes")
    print(' '.join(map(str, x)))

if __name__ == '__main__':
    main()