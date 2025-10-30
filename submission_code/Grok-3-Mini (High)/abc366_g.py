import sys

def find_null_space(A):
    N = len(A)
    A_copy = [row[:] for row in A]  # Create a copy of the matrix
    pivot_cols = set()
    for col in range(N):
        pivot_row = -1
        for row in range(col, N):
            if A_copy[row][col] == 1:
                pivot_row = row
                break
        if pivot_row != -1:
            # Swap rows
            A_copy[col], A_copy[pivot_row] = A_copy[pivot_row], A_copy[col]
            # Eliminate in other rows
            for row in range(N):
                if row != col and A_copy[row][col] == 1:
                    for k in range(N):
                        A_copy[row][k] ^= A_copy[col][k]
            pivot_cols.add(col)
    free_vars = [j for j in range(N) if j not in pivot_cols]
    basis = []
    for f in free_vars:
        X = [0] * N
        X[f] = 1  # Set the free variable to 1
        # Set pivot variables in decreasing order
        pivot_list = sorted(list(pivot_cols), reverse=True)
        for p in pivot_list:
            sum_xor = 0
            for k in range(p + 1, N):
                if A_copy[p][k] == 1:
                    sum_xor ^= X[k]
            X[p] = sum_xor
        basis.append(X)
    return basis

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2

# Build adjacency matrix
A = [[0 for _ in range(N)] for _ in range(N)]
for _ in range(M):
    u = int(data[index]) - 1  # Convert to 0-based
    v = int(data[index + 1]) - 1  # Convert to 0-based
    index += 2
    A[u][v] = 1
    A[v][u] = 1  # Ensure symmetry

# Find basis of null space
basis = find_null_space(A)

# Check if all vertices are covered
all_covered = True
for v in range(N):
    covered = False
    for vec in basis:
        if vec[v] == 1:
            covered = True
            break
    if not covered:
        all_covered = False
        break

if not all_covered:
    print("No")
else:
    # Output Yes and compute assignments
    print("Yes")
    num_basis = len(basis)
    values = []
    for v in range(N):
        val = 0
        for b in range(num_basis):
            if basis[b][v] == 1:
                val |= (1 << b)  # Set the b-th bit if basis[b][v] is 1
        values.append(val)
    # Output the values space-separated
    print(' '.join(map(str, values)))