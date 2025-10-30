import sys

sys.setrecursionlimit(2 * 10**5 + 500) # Increased recursion limit for deep DFS/Kosaraju's

def solve():
    N, M = map(int, sys.stdin.readline().split())
    testimonies = []
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        testimonies.append((A - 1, B - 1, C)) # 0-indexed villagers

    # Number of boolean variables for 2-SAT:
    # h_0 ... h_{N-1} (Honest/Liar status)
    # c_0 ... c_{N-1} (Confused/Not Confused status)
    # aux_0 ... aux_{M-1} (Auxiliary variables for XOR constraints)
    # Total variables: N (for h) + N (for c) + M (for aux) = 2*N + M
    num_vars = 2 * N + M

    # Each variable 'v' has two literals: 'v_true' (2*v) and 'v_false' (2*v + 1)
    # Graph nodes are 0 to 2*num_vars - 1
    num_nodes = 2 * num_vars
    
    adj = [[] for _ in range(num_nodes)]
    rev_adj = [[] for _ in range(num_nodes)]

    # Helper to add an implication u -> v (and its contrapositive ~v -> ~u)
    def add_implication(u_lit, v_lit):
        adj[u_lit].append(v_lit)
        adj[v_lit ^ 1].append(u_lit ^ 1) # v_lit ^ 1 is the complement of v_lit

    # Helper to add constraints for var1 XOR var2 = target_val
    # var_idx1, var_idx2 are variable indices (0 to num_vars-1)
    def add_xor_eq(var_idx1, var_idx2, target_val):
        lit1_T = 2 * var_idx1
        lit1_F = 2 * var_idx1 + 1
        lit2_T = 2 * var_idx2
        lit2_F = 2 * var_idx2 + 1

        if target_val == 0: # var1 == var2
            add_implication(lit1_T, lit2_T)
            add_implication(lit1_F, lit2_F)
        else: # var1 != var2
            add_implication(lit1_T, lit2_F)
            add_implication(lit1_F, lit2_T)

    # Map variable indices:
    # h_i (villager i's honest status): i
    # c_i (villager i's confused status): N + i
    # aux_j (auxiliary var for j-th testimony): 2*N + j

    for j, (A, B, C_val) in enumerate(testimonies):
        h_A_var = A
        h_B_var = B
        c_A_var = N + A
        aux_j_var = 2 * N + j # auxiliary variable for this testimony

        # Constraint 1: (h_A XOR h_B) = aux_j
        # This is (h_A XOR h_B XOR aux_j = 0)
        # It needs to be broken down into 2-SAT clauses.
        # It's (aux_j AND (h_A != h_B)) OR (~aux_j AND (h_A = h_B))
        # This gives two main implications:
        # 1. aux_j => (h_A != h_B)
        #    This is ~aux_j OR (h_A != h_B)
        #    Clauses: (~aux_j OR h_A OR h_B) AND (~aux_j OR ~h_A OR ~h_B)
        #    Equivalent implications:
        #    aux_j_T -> h_A_F, aux_j_T -> h_B_F  (wrong directly)
        # Correct 2-SAT formulation for P => (X XOR Y = K):
        # (P \land X \land Y) \implies (K=0)  (if P=T, X=T, Y=T then K must be 0)
        # (P \land X \land 
eg Y) \implies (K=1)  (if P=T, X=T, Y=F then K must be 1)
        # ... and so on for all 8 combinations.
        # A simpler way for (X XOR Y) = Z is 4 pairs of 2-SAT clauses:
        # X=0, Y=0 => Z=0   (X=F, Y=F => Z=F)  -> (X_F -> Z_F), (Y_F -> Z_F)
        # X=0, Y=1 => Z=1   (X=F, Y=T => Z=T)  -> (X_F -> Z_T), (Y_T -> Z_T)
        # X=1, Y=0 => Z=1   (X=T, Y=F => Z=T)  -> (X_T -> Z_T), (Y_F -> Z_T)
        # X=1, Y=1 => Z=0   (X=T, Y=T => Z=F)  -> (X_T -> Z_F), (Y_T -> Z_F)

        # Let's map this properly. Each of the above lines means: (X OR Y OR Z) etc.
        # This translates to:
        # h_A XOR h_B XOR aux_j = 0
        # If any two are equal, the third must be 0. If any two are unequal, the third must be 1.
        # This gives 8 implications (and their contrapositives):
        # (h_A and h_B) => aux_j_false   (if h_A=T, h_B=T, then aux_j must be F)
        add_implication(2*h_A_var, (2*aux_j_var)^1) # h_A_T -> aux_j_F (if h_B is also T, then h_A=h_B)
        add_implication(2*h_B_var, (2*aux_j_var)^1) # h_B_T -> aux_j_F

        add_implication(2*h_A_var^1, (2*aux_j_var)^1) # h_A_F -> aux_j_F
        add_implication(2*h_B_var^1, (2*aux_j_var)^1) # h_B_F -> aux_j_F
        
        # This is for (h_A = h_B) => (aux_j = 0)
        add_implication(2*h_A_var, 2*h_B_var^1) # if h_A is true, h_B is false
        add_implication(2*h_B_var, 2*h_A_var^1)
        add_implication(2*h_A_var^1, 2*h_B_var)
        add_implication(2*h_B_var^1, 2*h_A_var)

        # Actually, for X XOR Y XOR Z = K, use the transformation:
        # Create a helper for `add_equiv_pair(lit1, lit2, is_same)`
        # This is the 2-SAT transformation for (X = Y) or (X = ~Y)
        def add_equiv_pair(lit1, lit2, is_same): # is_same=True for X=Y, False for X!=Y
            if is_same: # X=Y
                add_implication(lit1, lit2)
                add_implication(lit1^1, lit2^1)
            else: # X!=Y
                add_implication(lit1, lit2^1)
                add_implication(lit1^1, lit2)

        # (h_A XOR h_B) = aux_j_val
        # This means:
        # aux_j_val is true (2*aux_j_var) => (h_A != h_B)
        add_equiv_pair(2*h_A_var, 2*h_B_var, False)
        add_implication((2*aux_j_var)^1, 2*h_A_var) # if aux_j_val is false, h_A and h_B must be equal
        add_implication((2*aux_j_var)^1, 2*h_B_var)

        # aux_j_val is false (2*aux_j_var + 1) => (h_A = h_B)
        add_equiv_pair(2*h_A_var, 2*h_B_var, True)
        add_implication(2*aux_j_var, 2*h_A_var^1) # if aux_j_val is true, h_A and h_B must be different
        add_implication(2*aux_j_var, 2*h_B_var^1)

        # This is the correct set of implications for (h_A XOR h_B) = aux_j.
        # This generates 8 implications. (4 for aux_j=T, 4 for aux_j=F)

        # Constraint 2: (aux_j XOR c_A) = C_val
        # This means:
        # C_val=0 => (aux_j = c_A)
        # C_val=1 => (aux_j != c_A)
        add_xor_eq(aux_j_var, c_A_var, C_val) # This adds 4 implications correctly


    # Kosaraju's algorithm for SCCs
    visited = [False] * num_nodes
    order = []
    
    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(num_nodes):
        if not visited[i]:
            dfs1(i)

    scc_id = [-1] * num_nodes
    scc_count = 0

    def dfs2(u):
        scc_id[u] = scc_count
        for v in rev_adj[u]:
            if scc_id[v] == -1:
                dfs2(v)

    order.reverse()
    for u in order:
        if scc_id[u] == -1:
            dfs2(u)
            scc_count += 1

    # Check for contradictions
    for i in range(num_vars):
        if scc_id[2*i] == scc_id[2*i+1]:
            sys.stdout.write("-1
")
            return

    # Construct a satisfying assignment
    assignment = [False] * num_vars # Default to false for now

    # A literal is true if its SCC comes after its complement's SCC in topological order (of SCCs).
    # This means if scc_id(lit) < scc_id(lit^1), then lit should be true.
    # Because scc_id are assigned in reverse topological order.
    for i in range(num_vars):
        if scc_id[2*i] < scc_id[2*i+1]:
            assignment[i] = True
        else:
            assignment[i] = False

    # Output the confused status string
    result = ['0'] * N
    for i in range(N):
        if assignment[N + i]: # If c_i is true (variable N+i is true)
            result[i] = '1'
    
    sys.stdout.write("".join(result) + "
")

# Run the solver
solve()