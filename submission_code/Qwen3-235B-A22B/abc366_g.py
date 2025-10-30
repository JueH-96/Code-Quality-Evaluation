def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input[ptr]) - 1
        ptr += 1
        v = int(input[ptr]) - 1
        ptr += 1
        adj[u].append(v)
        adj[v].append(u)
    
    # Build equations for each node with degree >= 1
    equations = []
    for v in range(N):
        if len(adj[v]) == 0:
            continue
        mask = 0
        for u in adj[v]:
            mask ^= (1 << u)
        equations.append(mask)
    
    # Perform Gaussian elimination
    N_VARS = N
    mat = equations.copy()
    rank = 0
    rows = len(mat)
    if rows == 0:
        # Null space is entire space
        print("Yes")
        X = [1] * N
        print(' '.join(map(str, X)))
        return
    
    # Gaussian elimination in GF(2)
    cur_row = 0
    for col in range(N_VARS):
        pivot = -1
        for r in range(cur_row, rows):
            if (mat[r] >> col) & 1:
                pivot = r
                break
        if pivot == -1:
            continue
        mat[cur_row], mat[pivot] = mat[pivot], mat[cur_row]
        for r in range(rows):
            if r != cur_row and ((mat[r] >> col) & 1):
                mat[r] ^= mat[cur_row]
        cur_row += 1
    rank = cur_row
    
    # Find pivot columns
    pivot_cols = []
    prow = 0
    pivots = {}
    for col in range(N_VARS):
        if prow < rank and (mat[prow] >> col) & 1:
            pivot_cols.append(col)
            pivots[col] = prow
            prow += 1
        else:
            continue
    
    # Find free variables
    free = []
    for col in range(N_VARS):
        if col not in pivots:
            free.append(col)
    
    # If no free variables, null space is trivial (only zero)
    if len(free) == 0:
        print("No")
        return
    
    # Build null space basis
    basis = []
    for f_var in free:
        vec = 0
        vec |= (1 << f_var)
        # Solve for pivot variables
        for pr in range(rank):
            row = pr
            row_mask = mat[row]
            pc = -1
            for c in range(N_VARS):
                if (row_mask >> c) & 1:
                    pc = c
                    break
            # Equation: row_mask * x = 0 mod 2
            # sum over variables in row_mask
            # x_pc = sum_{other vars} x_j * bit_j
            sum_val = 0
            # Iterate all variables except pc
            mask_without_pc = row_mask ^ (1 << pc)
            # Check if f_var is set in mask_without_pc
            if (mask_without_pc >> f_var) & 1:
                sum_val ^= 1
            # Check other free variables (but since we set them to 0 except f_var)
            # they contribute 0
            # Set x_pc to sum_val
            if sum_val == 1:
                vec |= (1 << pc)
        basis.append(vec)
    
    # Check if all variables are covered in the basis vectors
    OR_mask = 0
    for vec in basis:
        OR_mask |= vec
    if OR_mask != ( (1 << N_VARS) - 1 ):
        print("No")
        return
    
    # Build X[i]
    X = [0]*N_VARS
    for i in range(N_VARS):
        for k in range(len(basis)):
            if (basis[k] >> i) & 1:
                X[i] |= (1 << k)
    # Ensure that X[i] is at least 1
    for x in X:
        if x == 0:
            print("No")
            return
    
    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()