# YOUR CODE HERE
def main():
    import sys, random
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return

    # Build the undirected graph (0-indexed)
    graph = [set() for _ in range(n)]
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        graph[u].add(v)
        graph[v].add(u)
        
    # Compute degrees and immediately reject if any vertex with degree exactly 1 exists.
    deg = [len(graph[i]) for i in range(n)]
    for i in range(n):
        if deg[i] == 1:
            sys.stdout.write("No")
            return

    # Partition the vertices into connected components (for vertices with edges)
    comp_id = [-1] * n
    comps = []
    visited = [False] * n
    def dfs(u, cid, comp_list):
        stack = [u]
        visited[u] = True
        comp_id[u] = cid
        comp_list.append(u)
        while stack:
            cur = stack.pop()
            for nb in graph[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    comp_id[nb] = cid
                    comp_list.append(nb)
                    stack.append(nb)
    cid = 0
    for i in range(n):
        if deg[i] > 0 and not visited[i]:
            comp_list = []
            dfs(i, cid, comp_list)
            comps.append(comp_list)
            cid += 1

    # Prepare global assignment array; for isolated vertices (deg 0) we assign 1.
    assign = [0] * n
    for i in range(n):
        if deg[i] == 0:
            assign[i] = 1

    # Helper function: compute a basis for the nullspace (kernel) mod2.
    # We represent a row (or vector) as an integer with k bits (LSB = column 0).
    def compute_nullspace_basis(mat, k):
        A = mat[:]  # copy of matrix rows (each an integer in [0, 2^k))
        pivot_of_row = [-1] * k  # pivot column for each pivot row
        r_idx = 0
        for col in range(k):
            pivot = -1
            for r in range(r_idx, k):
                if (A[r] >> col) & 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            A[r_idx], A[pivot] = A[pivot], A[r_idx]
            pivot_of_row[r_idx] = col
            # Eliminate the col-bit from all other rows
            for r in range(k):
                if r != r_idx and ((A[r] >> col) & 1):
                    A[r] ^= A[r_idx]
            r_idx += 1
            if r_idx == k:
                break
        rank = r_idx
        d = k - rank  # nullspace dimension
        if d <= 0:
            return []  # only trivial solution exists
        pivot_cols = set()
        for i in range(rank):
            pivot_cols.add(pivot_of_row[i])
        free_cols = [c for c in range(k) if c not in pivot_cols]
        
        basis = []
        # For each free column, set that free variable to 1 and others (free) to 0,
        # then solve for pivot variables.
        for free in free_cols:
            x = 0
            x |= (1 << free)
            for i in range(rank - 1, -1, -1):
                pc = pivot_of_row[i]
                s = 0
                for fc in free_cols:
                    if (A[i] >> fc) & 1:
                        s ^= ((x >> fc) & 1)
                if s:
                    x |= (1 << pc)
                else:
                    x &= ~(1 << pc)
            basis.append(x)
        return basis

    # Process each non-isolated component.
    for comp in comps:
        k = len(comp)
        # Build the k x k binary (GF2) matrix for the component.
        # Order: list 'comp' is the list of vertex indices.
        # Let row i correspond to vertex comp[i]. For j from 0 to k-1, set bit j if comp[i] is adjacent to comp[j].
        mat = [0] * k
        pos = {}
        for i, v in enumerate(comp):
            pos[v] = i
        for i, v in enumerate(comp):
            row = 0
            for nb in graph[v]:
                if nb in pos:
                    j = pos[nb]
                    row |= (1 << j)
            mat[i] = row
        
        basis = compute_nullspace_basis(mat, k)
        d = len(basis)
        if d == 0:
            sys.stdout.write("No")
            return
        # Check that for each vertex (coordinate) there is at least one kernel vector with a 1.
        for i in range(k):
            found = False
            for vec in basis:
                if ((vec >> i) & 1):
                    found = True
                    break
            if not found:
                sys.stdout.write("No")
                return

        # If the kernel is 1-dimensional, then every solution is constant.
        # In GF2 the sole nonzero solution is the constant vector (all ones).
        if d == 1:
            vec = basis[0]
            if vec != (1 << k) - 1:
                sys.stdout.write("No")
                return
            # Constant solution gives every vertex the same 60-bit number.
            # But for vertex v the equation becomes (X repeated deg(v) times) = X if deg(v) odd, else 0.
            # So every vertex must have even degree.
            if any(deg[v] % 2 for v in comp):
                sys.stdout.write("No")
                return
            const_val = (1 << 60) - 1
            for v in comp:
                assign[v] = const_val
        else:
            # d >= 2; we have freedom.
            # We now choose 60 independent random solutions (each a nonzero element of the nullspace)
            # to represent 60 bits. For each vertex (coordinate in the kernel) the j-th bit
            # comes from the j-th chosen solution.
            comp_val = [0] * k
            ok = False
            for _ in range(20):
                cand = [0] * k
                for bit in range(60):
                    # Pick a random nonzero kernel element.
                    r = random.randrange(1, (1 << d))
                    vec_bit = 0
                    for j in range(d):
                        if (r >> j) & 1:
                            vec_bit ^= basis[j]
                    for i in range(k):
                        if (vec_bit >> i) & 1:
                            cand[i] |= (1 << bit)
                if all(val != 0 for val in cand):
                    comp_val = cand
                    ok = True
                    break
            if not ok:
                sys.stdout.write("No")
                return
            for i, v in enumerate(comp):
                assign[v] = comp_val[i]
                
    # Finally, output the answer.
    out = []
    out.append("Yes")
    out.append(" ".join(str(x) for x in assign))
    sys.stdout.write("
".join(out))
    
if __name__ == '__main__':
    main()