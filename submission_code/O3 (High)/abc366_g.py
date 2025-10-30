import sys

# ------------------------------------------------------------
# Gaussian elimination over GF(2) that also returns a basis of
# the null–space (kernel) of the given n×n matrix.
# Every row of the matrix is stored as an int whose j-th bit is
# the element in column j (0 ≤ j < n).
# ------------------------------------------------------------
def kernel_basis(n: int, rows):
    rows = rows[:]                       # make a local copy
    pivot_row_of_col = [-1] * n          # pivot position for every column
    rank = 0

    # convert the matrix to (reduced) row-echelon form
    for col in range(n):
        sel = -1
        for i in range(rank, n):
            if (rows[i] >> col) & 1:
                sel = i
                break
        if sel == -1:
            continue                     # no pivot in this column
        rows[rank], rows[sel] = rows[sel], rows[rank]
        pivot_row_of_col[col] = rank

        # eliminate the pivot column from every other row
        for i in range(n):
            if i != rank and ((rows[i] >> col) & 1):
                rows[i] ^= rows[rank]
        rank += 1

    # free columns = null-space dimension
    free_cols = [c for c in range(n) if pivot_row_of_col[c] == -1]

    basis = []
    for f in free_cols:
        vec = 1 << f                     # this free variable is 1
        # determine the pivot variables implied by this choice
        for col, r in enumerate(pivot_row_of_col):
            if r != -1 and ((rows[r] >> f) & 1):
                vec |= 1 << col
        basis.append(vec)

    return basis


# ------------------------------------------------------------
def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    adj = [0] * N                        # adjacency rows (as bitsets)

    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    # obtain a basis of the kernel of the adjacency matrix
    basis = kernel_basis(N, adj)

    # if the kernel is trivial, no assignment except all zeros exists
    if not basis:
        print("No")
        return

    # check that every vertex appears in at least one basis vector
    cover = 0
    for vec in basis:
        cover |= vec
    if cover != (1 << N) - 1:
        print("No")
        return

    # build the numbers to write on the vertices
    # bit j of the number corresponds to basis[j]
    X = [0] * N
    for bit_idx, vec in enumerate(basis):
        bit_val = 1 << bit_idx           # ≤ 2^59 (N ≤ 60)
        for v in range(N):
            if (vec >> v) & 1:
                X[v] |= bit_val

    # all X_v are now positive and not greater than 2^60-1
    print("Yes")
    print(' '.join(str(x) for x in X))


if __name__ == "__main__":
    main()