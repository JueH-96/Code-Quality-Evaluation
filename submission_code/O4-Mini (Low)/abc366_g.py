import sys

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0]); M = int(data[1])
    edges = []
    idx = 2
    adj = [[] for _ in range(N)]
    for i in range(M):
        u = int(data[idx]) - 1; v = int(data[idx+1]) - 1
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
    # If no edges, any assignment works:
    if M == 0:
        print("Yes")
        print(" ".join(["1"] * N))
        return

    # Build system: for each vertex v with deg>=1, sum of neighbor vars = 0 (mod2)
    # We'll build a list of equations over GF2: each is an int mask of length N.
    rows = []
    for v in range(N):
        if len(adj[v]) == 0: continue
        mask = 0
        for u in adj[v]:
            mask |= 1 << u
        rows.append(mask)

    # Gaussian elimination mod2 to find nullspace basis
    # We'll do elimination on 'rows' (list of ints), columns 0..N-1
    rows = rows[:]  # copy
    R = len(rows)
    pivot_col = [-1] * N  # pivot_col[c] = row index where c is pivot, or -1
    r = 0
    for c in range(N):
        # find row >= r with bit c set
        sel = -1
        for i in range(r, R):
            if (rows[i] >> c) & 1:
                sel = i
                break
        if sel == -1:
            continue
        # swap sel into row r
        rows[r], rows[sel] = rows[sel], rows[r]
        pivot_col[c] = r
        # eliminate bit c from other rows
        for i in range(R):
            if i != r and ((rows[i] >> c) & 1):
                rows[i] ^= rows[r]
        r += 1
        if r == R:
            break

    # determine free variables
    free_vars = [c for c in range(N) if pivot_col[c] == -1]
    d = len(free_vars)
    if d == 0:
        print("No")
        return

    # Build nullspace basis vectors b_j for each free var f_j
    # Each basis vector is length N, stored as int mask
    basis = []
    for fj in free_vars:
        vec = 1 << fj
        # back-substitute pivot variables
        for c in range(N):
            pr = pivot_col[c]
            if pr != -1:
                # if row pr has bit fj, then pivot var c = 1 in this solution
                if (rows[pr] >> fj) & 1:
                    vec |= 1 << c
        basis.append(vec)

    # We only can use up to 60 bits (positions), so truncate if needed
    if len(basis) > 60:
        basis = basis[:60]
        d = 60

    # Check that every vertex v is nonzero in at least one basis vector
    # so that its final integer X_v will be > 0
    covered = [False] * N
    for j in range(d):
        vec = basis[j]
        for v in range(N):
            if (vec >> v) & 1:
                covered[v] = True
    for v in range(N):
        # vertices with deg==0 are fine; they get 1 later if needed
        if len(adj[v]) > 0 and not covered[v]:
            print("No")
            return

    # Construct X_v as sum over j of (bit j)*2^j if basis[j] has bit at v
    X = [0] * N
    for v in range(N):
        val = 0
        for j in range(d):
            if (basis[j] >> v) & 1:
                val |= 1 << j
        if val == 0:
            # isolated vertex or no support: give it 1
            val = 1
        X[v] = val

    # All X[v] >= 1 by construction
    print("Yes")
    print(" ".join(str(x) for x in X))

if __name__ == "__main__":
    main()