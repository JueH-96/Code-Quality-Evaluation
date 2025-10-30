import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # Build adjacency matrix mod 2
    mat = [0] * n
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        # set edge bits
        mat[u] |= (1 << v)
        mat[v] |= (1 << u)
    # We will row‐reduce this matrix to RREF to find null‐space of A x = 0
    # Make a copy because we'll destroy mat in elimination
    A = mat[:]  # rows A[i] is bitmask of row i
    nvar = n
    pivot_cols = []
    used_pivot = [False] * nvar
    row = 0
    for col in range(nvar):
        # find a row >= row with a 1 in column col
        sel = -1
        for r in range(row, nvar):
            if (A[r] >> col) & 1:
                sel = r
                break
        if sel < 0:
            # no pivot in this column => free variable
            continue
        # swap to bring pivot to current row
        A[row], A[sel] = A[sel], A[row]
        pivot_cols.append(col)
        used_pivot[col] = True
        # eliminate this column from all other rows
        mask = 1 << col
        for r in range(nvar):
            if r != row and ((A[r] & mask) != 0):
                A[r] ^= A[row]
        row += 1
        if row == nvar:
            break
    rank = row
    # Collect free variables
    free_vars = [c for c in range(nvar) if not used_pivot[c]]
    D = len(free_vars)
    # If no null‐space (only trivial solution), no nonzero assignment
    if D == 0:
        print("No")
        return
    # Build basis of null‐space: one basis vector per free variable
    basis = []
    for f in free_vars:
        # x[f] = 1, and for each pivot col p we set x[p] = A[row_of_pivot][f]
        vec = 1 << f
        # pivot_cols[k] is the pivot column of row k
        for k, pcol in enumerate(pivot_cols):
            if ((A[k] >> f) & 1):
                vec |= (1 << pcol)
        basis.append(vec)
    # Check that every vertex v is covered by at least one basis vector
    covered = [False] * nvar
    for vec in basis:
        # mark all bits where vec has a 1
        # we can iterate bits directly
        vmask = vec
        while vmask:
            lsb = vmask & -vmask
            v = (lsb.bit_length() - 1)
            covered[v] = True
            vmask ^= lsb
    if not all(covered):
        # some vertex is always zero in any null‐space vector => no positive number
        print("No")
        return
    # We can assign each vertex v the integer whose binary repr
    # has bit i = 1 iff basis[i][v] = 1.
    X = [0] * nvar
    # basis[i] is an int whose bit v says if x_v has bit i
    for v in range(nvar):
        xv = 0
        for i, vec in enumerate(basis):
            if ((vec >> v) & 1):
                xv |= (1 << i)
        # xv is nonzero by coverage check
        X[v] = xv
    # Output
    out = []
    out.append("Yes")
    # convert X to strings
    out.append(" ".join(str(x) for x in X))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()