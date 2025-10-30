# YOUR CODE HERE
import sys

def main():
    import sys, math
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    edges = [[] for _ in range(N)]
    for _ in range(M):
        u = int(data[idx])-1; idx +=1
        v = int(data[idx])-1; idx +=1
        edges[u].append(v)
        edges[v].append(u)
    # Rows: for each vertex with degree >=1, a bitmask of its neighbors
    equations = []
    for v in range(N):
        if len(edges[v]) >=1:
            mask = 0
            for u in edges[v]:
                mask |= (1 << u)
            equations.append(mask)
    M_eq = len(equations)
    # Perform Gaussian elimination over GF2
    basis = []
    row = 0
    for col in range(N):
        pivot = -1
        for r in range(row, M_eq):
            if (equations[r] >> col) &1:
                pivot = r
                break
        if pivot == -1:
            continue
        equations[row], equations[pivot] = equations[pivot], equations[row]
        for r in range(M_eq):
            if r != row and ((equations[r] >> col) &1):
                equations[r] ^= equations[row]
        basis.append(1 << col)
        row +=1
    rank = row
    null_dim = N - rank
    # If null_dim ==0, only solution is X=0, which is invalid since X_v >=1
    if null_dim ==0:
        # Check if X=0 is the only solution
        # But need X_v >=1, which is impossible
        print("No")
        return
    # Find basis for nullspace
    # Initialize free variables
    # The variables not in basis can be freely set
    pivot_cols = []
    for b in basis:
        for i in range(N):
            if (b >> i) &1:
                pivot_cols.append(i)
                break
    free_vars = [i for i in range(N) if i not in pivot_cols]
    # Generate nullspace basis vectors
    null_basis = []
    for fv in free_vars:
        vec = 1 << fv
        for i, pc in enumerate(pivot_cols):
            if (basis[i] >> fv) &1:
                vec ^= basis[i]
        null_basis.append(vec)
    # Now we have a nullspace basis of size null_dim
    # We need to select a subset of null_basis vectors to cover all N vertices
    # Greedy set cover
    uncovered = set(range(N))
    selected = []
    for i in range(len(null_basis)):
        if not uncovered:
            break
        best = -1
        best_idx = -1
        for j, vec in enumerate(null_basis):
            cover = 0
            for v in range(N):
                if ((vec >> v) &1) and v in uncovered:
                    cover +=1
            if cover > best:
                best = cover
                best_idx = j
        if best <=0:
            break
        selected.append(null_basis[best_idx])
        for v in range(N):
            if ((null_basis[best_idx] >> v) &1):
                uncovered.discard(v)
    if uncovered:
        print("No")
        return
    # Now assign each selected vector to a bit
    num_bits = len(selected)
    if num_bits >60:
        print("No")
        return
    X = [0]*N
    for bit, vec in enumerate(selected):
        for v in range(N):
            if ((vec >> v)&1):
                X[v] |= (1 << bit)
    # Now assign remaining bits to arbitrary
    for bit in range(num_bits,60):
        # Assign no additional bits
        pass
    # Check all X_v >=1
    valid = all(x >=1 for x in X)
    if not valid:
        print("No")
        return
    # Now, output
    print("Yes")
    print(' '.join(str(x) for x in X))

if __name__ == "__main__":
    main()