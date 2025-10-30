import sys

# The Gaussian elimination function modified to find RREF and null space basis
def gaussian_eliminate_and_find_null_basis(matrix_rows, n):
    """
    Performs Gaussian elimination over F_2 on a matrix represented by integer rows.
    Returns the rank and a basis for the null space.
    matrix_rows: list of integers, each representing a row.
    n: number of columns.
    """
    mat = list(matrix_rows) # Work on a copy
    rows = len(mat)

    # Phase 1: Forward elimination to get row echelon form and identify pivot columns
    pivot_cols = []
    current_row = 0
    for col in range(n):
        # Find a row with a 1 in this column (at or below current_row)
        swap_row = current_row
        while swap_row < rows and not (mat[swap_row] >> (n - 1 - col)) & 1:
            swap_row += 1

        if swap_row < rows:
            # Found pivot. Swap row into current_row.
            mat[current_row], mat[swap_row] = mat[swap_row], mat[current_row]

            # Eliminate below
            for i in range(rows):
                if i != current_row and (mat[i] >> (n - 1 - col)) & 1:
                    mat[i] ^= mat[current_row]

            pivot_cols.append(col)
            current_row += 1
        # If no pivot in this column, current_row stays the same, move to next column

    rank = len(pivot_cols)

    # Phase 1.5: Backward elimination to get Reduced Row Echelon Form (RREF)
    # `mat` is currently in Row Echelon Form (REF).
    # The first `rank` rows are the non-zero rows.
    # Iterate upwards from the last pivot row.
    for r_idx in range(rank - 1, -1, -1):
        pivot_c = pivot_cols[r_idx]
        # Eliminate above
        for i in range(r_idx - 1, -1, -1):
            if (mat[i] >> (n - 1 - pivot_c)) & 1:
                mat[i] ^= mat[r_idx]

    # Now `mat` contains the RREF in its first `rank` rows.
    # The non-zero rows (first `rank` rows) form a basis for the row space in RREF.

    free_cols = [j for j in range(n) if j not in pivot_cols]
    null_space_basis = []

    # Phase 2: Construct null space basis from RREF
    # Iterate through each free variable (corresponding to a free column)
    for k in range(len(free_cols)):
        basis_vector = [0] * n # Initialize basis vector for this free variable
        current_free_col = free_cols[k]

        # Set the free variable corresponding to this basis vector to 1
        basis_vector[current_free_col] = 1

        # Determine values for pivot variables using the RREF equations
        # Iterate through pivot rows (the first `rank` rows in the RREF)
        for r_idx in range(rank):
            pivot_c = pivot_cols[r_idx]
            # In RREF, row r_idx has a leading 1 at pivot_c.
            # The equation from this row implies x_{pivot_c} = sum_{j in free_cols} (coefficient of x_j) * x_j
            # For this specific basis vector, only x_{current_free_col} is 1 among free vars.
            # So, x_{pivot_c} = (coefficient of x_{current_free_col} in row r_idx) * 1
            # The coefficient is the bit at column current_free_col in mat[r_idx].
            # Bit (n-1-j) corresponds to column j.
            if (mat[r_idx] >> (n - 1 - current_free_col)) & 1:
                basis_vector[pivot_c] = 1 # Set the corresponding pivot position to 1

        null_space_basis.append(basis_vector)

    return rank, null_space_basis

def solve():
    N, M = map(int, sys.stdin.readline().split())

    adj_matrix_rows = [0] * N # Represent each row as an integer
    degrees = [0] * N

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        adj_matrix_rows[u] |= (1 << v)
        adj_matrix_rows[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1

    # Vertices with degree >= 1 impose constraints
    v_prime_indices = [i for i in range(N) if degrees[i] > 0]

    if not v_prime_indices:
        # No vertex with degree >= 1. The condition is vacuously true for all such vertices.
        # We just need to assign integers between 1 and 2^60-1.
        # Assigning 1 to all vertices works.
        print("Yes")
        print(*[1] * N)
        return

    # Construct the matrix A' containing rows of the adjacency matrix
    # for vertices with degree >= 1. This is the matrix of the system A'x = 0.
    a_prime_matrix_rows = [adj_matrix_rows[i] for i in v_prime_indices]

    # Find the rank and a basis for the null space of A'
    rank, null_basis = gaussian_eliminate_and_find_null_basis(a_prime_matrix_rows, N)

    d = N - rank # Dimension of the null space

    if d == 0:
        # The null space only contains the zero vector.
        # This means the only solution to A'x = 0 is x = 0.
        # So, for any bit position k, x_{v, k} must be 0 for all v.
        # This would make X_v = 0 for all v, which is not allowed (must be >= 1).
        print("No")
        return

    # If d > 0, there are non-trivial solutions in the null space.
    # We need to check if there is a solution X_v >= 1 for all v.
    # A solution X_v >= 1 for all v exists if and only if for every vertex v,
    # there is at least one vector x in the null space such that x_v = 1.
    # This is equivalent to checking if for every column v (0 to N-1),
    # at least one basis vector in null_basis has a 1 at index v.
    # Equivalently, check if any column v in the basis matrix (d x N) is all zeros.

    basis_matrix_cols_are_nonzero = [False] * N
    for basis_vec in null_basis: # null_basis is a list of d vectors, each length N
        for v in range(N):
            if basis_vec[v] == 1:
                basis_matrix_cols_are_nonzero[v] = True

    if not all(basis_matrix_cols_are_nonzero):
        # For some vertex v, x_v must be 0 for ANY solution x in the null space.
        # This means X_v must be 0 for any valid assignment using bits from the null space.
        # This violates the condition X_v >= 1.
        print("No")
        return

    # If we reached here, a solution exists.
    # We can construct a solution by assigning the i-th basis vector
    # to the i-th bit position (0-indexed).
    # X_v = sum_{i=0}^{d-1} (null_basis[i][v]) * 2^i

    X = [0] * N
    for v in range(N):
        for i in range(d):
            if null_basis[i][v] == 1:
                 X[v] |= (1 << i) # Set bit i for vertex v

    # The constructed values X[v] are guaranteed to be:
    # 1. Solutions to the linear system (by construction).
    # 2. >= 1 (because we checked that column v in basis matrix is non-zero).
    # 3. < 2^d (since we use d bits). Since d <= N <= 60, X[v] < 2^60.
    # The required range is [1, 2^60 - 1]. X[v] >= 1 is checked. X[v] < 2^60 is true.
    # If d < 60, then X[v] <= 2^d - 1 <= 2^59 - 1 < 2^60 - 1.
    # If d = 60 (which only happens if N=60 and rank=0), X[v] <= 2^60 - 1.
    # So X[v] is within the required range [1, 2^60 - 1].

    print("Yes")
    print(*X)

solve()