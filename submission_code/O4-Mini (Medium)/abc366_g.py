import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Build adjacency matrix over GF(2)
    mat = [0] * N
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        # undirected
        mat[u] |= (1 << v)
        mat[v] |= (1 << u)
    # We want to solve B x = 0 over GF(2), where B is the adjacency matrix.
    # Perform Gaussian elimination to RREF to find the nullspace basis.
    pos = [-1] * N  # pos[col] = row index of pivot for that column
    row = 0
    for col in range(N):
        # find a pivot in rows [row..N-1] with bit col set
        sel = -1
        for r in range(row, N):
            if (mat[r] >> col) & 1:
                sel = r
                break
        if sel == -1:
            # no pivot in this column => free variable
            continue
        # swap into position
        mat[row], mat[sel] = mat[sel], mat[row]
        pos[col] = row
        # eliminate in all other rows
        for r in range(N):
            if r != row and ((mat[r] >> col) & 1):
                mat[r] ^= mat[row]
        row += 1
    rank = row
    # Nullspace dimension
    free_cols = [c for c in range(N) if pos[c] == -1]
    d = len(free_cols)
    # If no free variables, only trivial solution x=0, which is invalid since we need x_i >= 1
    if d == 0:
        print("No")
        return
    # Build basis vectors of the nullspace. Each vector is length N.
    basis = []
    for j, fc in enumerate(free_cols):
        v = [0] * N
        # free variable fc is set to 1
        v[fc] = 1
        # pivot variables determined by RREF rows
        for p in range(N):
            pr = pos[p]
            if pr == -1:
                # this p is free too, skip
                continue
            # x[p] = sum_{q where mat[pr][q] = 1} x[q] over GF(2)
            # we only need to sum over free columns q
            rowmask = mat[pr]
            val = 0
            # sum contributions from free columns
            for q in free_cols:
                if (rowmask >> q) & 1:
                    val ^= v[q]
            v[p] = val
        basis.append(v)
    # Check that every vertex i is "covered": there is at least one basis vector with v[i] = 1
    covered = [False] * N
    for v in basis:
        for i in range(N):
            if v[i]:
                covered[i] = True
    if not all(covered):
        print("No")
        return
    # We can now assign for each vertex i a 60-bit integer whose j-th bit is basis[j][i].
    # Since d <= N <= 60, this fits in [1, 2^60-1]. And each vertex has at least one bit=1.
    X = [0] * N
    for i in range(N):
        xi = 0
        # set bit j if basis[j][i] = 1
        for j in range(d):
            if basis[j][i]:
                xi |= (1 << j)
        # xi >= 1 by coverage
        X[i] = xi
    # Output
    print("Yes")
    print(" ".join(str(x) for x in X))

if __name__ == "__main__":
    main()