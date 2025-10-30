# YOUR CODE HERE
import sys

def solve():
    """
    This function solves the problem by applying principles of linear algebra over GF(2).
    The main steps are:
    1. Preliminary check: A graph with any degree-1 vertex has no solution.
    2. Decompose the graph into connected components. Each component is solved independently.
    3. For each non-trivial component, check two conditions using its adjacency matrix A_c:
       a. A_c must be singular (rank < size), so a non-zero solution for bits exists.
       b. For every vertex v, its corresponding basis vector e_v must not be in the image of A_c.
          This ensures X_v can be made non-zero.
    4. If all checks pass, a solution is constructed using a basis of the kernel of A_c for each component.
    """

    # Using an iterative BFS for component finding avoids potential recursion depth issues.
    try:
        input = sys.stdin.readline
        N, M = map(int, input().split())
        adj = [[] for _ in range(N)]
        degrees = [0] * N
        if M > 0:
            for _ in range(M):
                u, v = map(int, input().split())
                u -= 1
                v -= 1
                adj[u].append(v)
                adj[v].append(u)
                degrees[u] += 1
                degrees[v] += 1
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    for i in range(N):
        if degrees[i] == 1:
            print("No")
            return

    # Find connected components using BFS
    visited = [False] * N
    components = []
    for i in range(N):
        if not visited[i]:
            comp = []
            q = [i]
            visited[i] = True
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                comp.append(u)
                for v_neighbor in adj[u]:
                    if not visited[v_neighbor]:
                        visited[v_neighbor] = True
                        q.append(v_neighbor)
            components.append(comp)

    # For each non-trivial component, perform linear algebra checks
    for comp in components:
        n_c = len(comp)
        if n_c <= 1:
            continue

        node_to_idx = {node: i for i, node in enumerate(comp)}
        
        A = [[0] * n_c for _ in range(n_c)]
        for u_orig in comp:
            u_local = node_to_idx[u_orig]
            for v_orig in adj[u_orig]:
                if v_orig in node_to_idx:
                    v_local = node_to_idx[v_orig]
                    A[u_local][v_local] = 1
        
        B = [row[:] for row in A]
        pivot_cols = []
        row_head = 0
        for col in range(n_c):
            if row_head < n_c:
                pivot_row = row_head
                while pivot_row < n_c and B[pivot_row][col] == 0:
                    pivot_row += 1
                
                if pivot_row < n_c:
                    B[row_head], B[pivot_row] = B[pivot_row], B[row_head]
                    for i in range(n_c):
                        if i != row_head and B[i][col] == 1:
                            for k in range(col, n_c):
                                B[i][k] ^= B[row_head][k]
                    pivot_cols.append(col)
                    row_head += 1
        rank = row_head
        RREF = B

        if rank == n_c:
            print("No")
            return

        row_basis = RREF[:rank]
        for i in range(n_c):
            e_i_T = [0] * n_c
            e_i_T[i] = 1
            
            temp_vec = e_i_T[:]
            for r in range(rank):
                if temp_vec[pivot_cols[r]] == 1:
                    for c_idx in range(n_c):
                        temp_vec[c_idx] ^= row_basis[r][c_idx]

            is_in_span = all(x == 0 for x in temp_vec)
            if is_in_span:
                print("No")
                return

    # All checks passed, a solution exists. Construct it.
    print("Yes")
    
    X = [0] * N
    
    for comp in components:
        n_c = len(comp)
        if n_c == 0: continue

        if n_c == 1:
            X[comp[0]] = 1
            continue
            
        node_to_idx = {node: i for i, node in enumerate(comp)}
        idx_to_node = {i: node for i, node in enumerate(comp)}
        
        A = [[0] * n_c for _ in range(n_c)]
        for u_orig in comp:
            u_local = node_to_idx[u_orig]
            for v_orig in adj[u_orig]:
                if v_orig in node_to_idx:
                    v_local = node_to_idx[v_orig]
                    A[u_local][v_local] = 1
        
        B = [row[:] for row in A]
        pivot_cols = []
        row_head = 0
        for col in range(n_c):
            if row_head < n_c:
                pivot_row = row_head
                while pivot_row < n_c and B[pivot_row][col] == 0:
                    pivot_row += 1
                if pivot_row < n_c:
                    B[row_head], B[pivot_row] = B[pivot_row], B[row_head]
                    for i in range(n_c):
                        if i != row_head and B[i][col] == 1:
                            for k in range(col, n_c):
                                B[i][k] ^= B[row_head][k]
                    pivot_cols.append(col)
                    row_head += 1
        rank = row_head
        RREF = B
        
        free_cols = [j for j in range(n_c) if j not in pivot_cols]
        dim_ker = len(free_cols)
        
        kernel_basis = []
        for free_col in free_cols:
            b = [0] * n_c
            b[free_col] = 1
            for i, piv_col in enumerate(pivot_cols):
                if RREF[i][free_col] == 1:
                    b[piv_col] = 1
            kernel_basis.append(b)

        for i in range(n_c):
            val = 0
            for j in range(dim_ker):
                if kernel_basis[j][i] == 1:
                    val ^= (1 << j)
            X[idx_to_node[i]] = val

    print(*X)

if __name__ == "__main__":
    solve()