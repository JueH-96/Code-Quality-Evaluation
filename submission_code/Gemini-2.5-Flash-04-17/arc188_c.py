import sys

# Function for symmetric difference of sets
def symmetric_difference(set1, set2):
    # Uses Python's built-in set symmetric_difference which should be reasonably optimized
    return set1.symmetric_difference(set2)

# DSU for S graph
parent_S = []
s_const = [] # S_j ^ S_root = s_const ^ XOR_{k in c_deps} C_k
c_deps = [] # frozenset of indices k

def find_S(j):
    if parent_S[j] == j:
        return (j, 0, frozenset()) # (root, constant, frozenset of C dependencies)
    
    # Path compression updates the difference relative to the new parent (root)
    root, const_p, deps_p = find_S(parent_S[j])
    
    # s_const[j] stores S_j ^ S_parent_S[j], update to S_j ^ S_root
    s_const[j] = s_const[j] ^ const_p

    # c_deps[j] stores d_c(j, parent_S[j]), update to d_c(j, root)
    c_deps[j] = symmetric_difference(c_deps[j], deps_p)
    
    parent_S[j] = root # Path compression
    
    # Return value from function should be the difference relative to the found root
    return (root, s_const[j], c_deps[j])


# DSU for C variables
parent_C = []
diff_C = [] # C_j ^ C_root = diff_C

# Special node 0 represents value 0
def find_C(j):
    if parent_C[j] == j:
        return (j, 0) # (root, difference C_j ^ C_root)
    
    root, diff_p = find_C(parent_C[j])
    
    # diff_C[j] stores difference C_j ^ C_parent_C[j]
    # diff_p stores difference C_parent_C[j] ^ C_root
    # diff_C[j] ^ diff_p = (C_j ^ C_parent_C[j]) ^ (C_parent_C[j] ^ C_root) = C_j ^ C_root
    diff_C[j] = diff_C[j] ^ diff_p
    
    parent_C[j] = root
    return (root, diff_C[j])

# Union for C variables
# Represents constraint C_u ^ C_v = k
def union_C(u, v, k):
    r_u, d_u = find_C(u)
    r_v, d_v = find_C(v)
    
    if r_u != r_v:
        parent_C[r_v] = r_u
        # We want C_r_v ^ C_r_u = some_diff
        # C_u ^ C_r_u = d_u => C_u = C_r_u ^ d_u
        # C_v ^ C_r_v = d_v => C_v = C_r_v ^ d_v
        # C_u ^ C_v = k
        # (C_r_u ^ d_u) ^ (C_r_v ^ d_v) = k
        # C_r_u ^ C_r_v = k ^ d_u ^ d_v
        diff_C[r_v] = k ^ d_u ^ d_v
        return True
    else: # r_u == r_v
        # Check consistency
        # Existing C_u ^ C_v = (C_r_u ^ d_u) ^ (C_r_u ^ d_v) = d_u ^ d_v
        # Required C_u ^ C_v = k
        if k != (d_u ^ d_v):
            return False # Contradiction
        return True

# Use adjacency list to store testimonies
adj = [] # adj[u] is list of (v, A, B, T) for testimonies (A, B, T) where A=u or B=u

def solve():
    N, M = map(int, sys.stdin.readline().split())

    global parent_S, s_const, c_deps, parent_C, diff_C, adj

    adj = [[] for _ in range(N + 1)]
    # testimonies list is not strictly needed after building adj, but can be kept for clarity
    # testimonies = []
    for _ in range(M):
        A, B, T = map(int, sys.stdin.readline().split())
        # testimonies.append((A, B, T)) # Not used directly in BFS
        # Store (neighbor, source, target, value)
        # Edge (A, B) from testimony (A, B, T)
        adj[A].append((B, A, B, T))
        adj[B].append((A, A, B, T)) # Add reverse edge for undirected traversal


    # --- Step 1: Traverse S graph and collect constraints on C ---
    parent_S = list(range(N + 1))
    s_const = [0] * (N + 1)
    c_deps = [frozenset() for _ in range(N + 1)]

    # List to store constraints on C variables: (frozenset of indices, constant)
    c_constraints = []

    # visited_S: -1 unvisited, 0 visiting/queued, 1 visited/processed
    visited_S = [-1] * (N + 1)

    for i in range(1, N + 1):
        if visited_S[i] == -1:
            q = [i]
            visited_S[i] = 0 # Mark as queued

            head = 0
            while head < len(q):
                u = q[head]
                head += 1

                # Process all neighbors v reachable from u via a testimony
                for v, A_t, B_t, T_t in adj[u]:
                    # This is edge (u, v) derived from testimony (A_t, B_t, T_t)
                    # Constraint from this testimony: S_A_t ^ S_B_t = T_t ^ C_A_t

                    # Required S_v ^ S_start from u:
                    # S_v ^ S_start = (S_u ^ S_start) ^ (S_u ^ S_v)
                    # We know S_A_t ^ S_B_t = T_t ^ C_A_t.
                    # If A_t=u, B_t=v: S_u ^ S_v = T_t ^ C_u
                    # If A_t=v, B_t=u: S_v ^ S_u = T_t ^ C_v => S_u ^ S_v = T_t ^ C_v

                    # Get current state of u relative to its S-component root (effectively S_start of this BFS)
                    u_s_const_val, u_c_deps_set = s_const[u], c_deps[u] # These are diffs relative to S_start (root of this BFS component) because find_S is not used here. find_S is used during constraint processing.

                    # Required S_v ^ S_start = (S_u ^ S_start) ^ (S_u ^ S_v)
                    # (S_u ^ S_v) = T_t ^ C_{A_t}
                    # S_v ^ S_start = (u_s_const_val ^ XOR(u_c_deps_set) C_k) ^ (T_t ^ C_A_t)
                    # S_v ^ S_start = (u_s_const_val ^ T_t) ^ (XOR(u_c_deps_set) C_k ^ C_A_t)
                    new_s_const_v = u_s_const_val ^ T_t
                    new_c_deps_v = symmetric_difference(u_c_deps_set, frozenset([A_t]))


                    if visited_S[v] == -1:
                        visited_S[v] = 0 # Mark as queued
                        s_const[v] = new_s_const_v
                        c_deps[v] = new_c_deps_v
                        q.append(v)
                    elif visited_S[v] == 0 or visited_S[v] == 1:
                         # Cycle or path to already visited node `v`
                         # Existing S_v ^ S_start = s_const[v] ^ XOR(c_deps[v]) C_k
                         # Required S_v ^ S_start = new_s_const_v ^ XOR(new_c_deps_v) C_k
                         # Constraint: (s_const[v] ^ new_s_const_v) ^ XOR(c_deps[v] ^ new_c_deps_v) C_k = 0
                         K = s_const[v] ^ new_s_const_v
                         M = symmetric_difference(c_deps[v], new_c_deps_v)
                         if M or K: # Only add non-trivial constraints
                             c_constraints.append((M, K))

            # Mark all nodes in this component as visited (1)
            for node in q:
                visited_S[node] = 1 # Mark as fully processed


    # --- Step 2: Solve constraints on C variables using C-DSU ---
    # N + 1 nodes: 1..N for villagers, 0 for value 0
    parent_C = list(range(N + 1))
    diff_C = [0] * (N + 1) # C_j ^ C_root = diff_C[j]. Node 0 is virtual root for value 0

    # Handle constraints
    for M, K in c_constraints:
        # Constraint: XOR_{k in M} C_k = K
        
        # Find roots and diffs for all k in M
        roots_info = [] # List of (root, diff_C_k)
        # M contains villager indices {1, ..., N}
        
        for k in M:
             roots_info.append(find_C(k))
        
        # Calculate XOR sum of diffs relative to their ultimate roots
        diffs_xor_sum = 0
        for root, diff in roots_info:
            diffs_xor_sum ^= diff
        
        # Calculate RHS of the constraint on roots
        K_prime = K ^ diffs_xor_sum
        
        # Calculate roots involved and their parity
        root_counts = {}
        for root, diff in roots_info: # roots_info contains (r_k, d_k) for k in M, where r_k is ultimate root
            root_counts[root] = root_counts.get(root, 0) + 1
        
        roots_odd_parity = [root for root, count in root_counts.items() if count % 2 == 1]
        
        # Apply constraint on roots
        if len(roots_odd_parity) == 0:
            if K_prime == 1:
                print("-1") # Contradiction
                return
            # If K_prime == 0, constraint is trivially satisfied.
        elif len(roots_odd_parity) == 1:
            r_star = roots_odd_parity[0]
            # Constraint C_r_star = K_prime. Union r_star with node 0 (value 0)
            if not union_C(r_star, 0, K_prime):
                print("-1") # Contradiction
                return
        elif len(roots_odd_parity) == 2:
            r1, r2 = roots_odd_parity
            # Constraint C_r1 ^ C_r2 = K_prime
            if not union_C(r1, r2, K_prime):
                print("-1") # Contradiction
                return
        else:
            # $|R_{odd}| > 2$. Constraint links > 2 roots.
            # If the constraint involves roots from different C-DSU components,
            # it cannot be modeled by pairwise unions and implies no solution in this model.
            # If all roots in R_odd are in the same component, check consistency.
            
            # Check if all roots are in the same C-DSU component.
            first_root_of_odd = roots_odd_parity[0]
            common_root_val, diff_first_ultimate = find_C(first_root_of_odd) # common_root_val is the ultimate root

            all_in_same_c_component = True
            for i in range(1, len(roots_odd_parity)):
                 r = roots_odd_parity[i]
                 curr_root_val, _ = find_C(r)
                 if curr_root_val != common_root_val:
                     all_in_same_c_component = False
                     break

            if all_in_same_c_component:
                 # Roots are all in the same C-DSU component. Check consistency.
                 # Constraint: $\bigoplus_{r \in R_{odd}} C_r = K_prime$
                 # C_r = C_{common_root_val} ^ (diff_C from r to common_root_val)
                 
                 xor_sum_diffs_of_roots_to_common = 0
                 # Need diff relative to common_root_val, not the ultimate root
                 # find_C(r) gives diff_r where C_r ^ C_ultimate = diff_r
                 # find_C(common_root_val) gives diff_common where C_common_root_val ^ C_ultimate = diff_common
                 # C_r ^ C_common_root_val = diff_r ^ diff_common
                 
                 _, diff_common_ultimate = find_C(common_root_val) # Get the diff from the common root (which is the ultimate root) to itself (0)
                 # Wait, diff_common_ultimate from find_C(common_root_val) is C_common_root_val ^ C_ultimate,
                 # where ultimate is find_C(common_root_val)[0]. It should be common_root_val itself unless it's linked to 0.
                 # If common_root_val is the root of its own component: find_C(common_root_val) = (common_root_val, 0)
                 # If common_root_val is linked to 0: find_C(common_root_val) = (0, diff)
                 # Let's use the diff relative to the ultimate root.

                 # $\bigoplus_{r \in R_{odd}} C_r = K_prime$
                 # Substitute $C_r = C_{ultimate\_root} \oplus diff\_C[r]$
                 # $\bigoplus_{r \in R_{odd}} (C_{ultimate\_root} \oplus diff\_C[r]) = K_prime$
                 # $(len(R_{odd}) \pmod 2) C_{ultimate\_root} \oplus \bigoplus_{r \in R_{odd}} diff\_C[r] = K_prime$
                 
                 xor_sum_diffs_of_roots_to_ultimate = 0
                 for r in roots_odd_parity:
                     _, diff_r_ultimate = find_C(r) # diff_r_ultimate is C_r ^ C_ultimate_root
                     xor_sum_diffs_of_roots_to_ultimate ^= diff_r_ultimate

                 # Get the ultimate root's current state
                 ultimate_root = common_root_val # common_root_val is the ultimate root of this component
                 actual_ultimate_root_root_in_C, actual_ultimate_root_diff_in_C = find_C(ultimate_root) # Check if ultimate root is linked to 0

                 if len(roots_odd_parity) % 2 == 0: # |R_odd| is even >= 4
                      # Constraint implies 0 ^ xor_sum_diffs_of_roots_to_ultimate = K_prime
                      if xor_sum_diffs_of_roots_to_ultimate != K_prime:
                          print("-1") # Contradiction
                          return
                 else: # |R_odd| is odd >= 3
                      # Constraint implies C_{ultimate_root} ^ xor_sum_diffs_of_roots_to_ultimate = K_prime
                      implied_ultimate_root_val = K_prime ^ xor_sum_diffs_of_roots_to_ultimate

                      # Check if ultimate root is already fixed inconsistently relative to 0
                      if actual_ultimate_root_root_in_C == 0: # Ultimate root is already linked to 0
                          fixed_val = actual_ultimate_root_diff_in_C # C_ultimate_root = fixed_val relative to C_0=0
                          if fixed_val != implied_ultimate_root_val:
                              print("-1") # Contradiction
                              return
                      else: # Ultimate root is currently free, now fixed by this constraint
                           # Link it to 0 with the implied value
                          if not union_C(ultimate_root, 0, implied_ultimate_root_val):
                               # union_C handles check if already linked to 0 inconsistently
                              print("-1") # Contradiction
                              return

            else: # Roots are in different components. This constraint links > 2 components.
                 # The current DSU can only represent pairwise relations.
                 # If such a constraint exists and involves roots from > 2 *different* C-DSU components,
                 # it cannot be resolved by pairwise DSU unions, suggesting no solution exists in this model.
                 print("-1")
                 return


    # --- Step 3: Construct output based on C-DSU state ---
    result = ""
    for i in range(1, N + 1):
        root_i, diff_i = find_C(i)
        if root_i == 0:
            # C_i ^ C_0 = diff_i. Since C_0 = 0, C_i = diff_i
            result += str(diff_i)
        else:
            # C_i ^ C_root_i = diff_i. C_root_i is a free variable relative to 0.
            # We can choose C_root_i = 0. Then C_i = diff_i ^ C_root_i = diff_i ^ 0 = diff_i.
            # Any consistent choice for free roots is fine. Choosing 0 is simple.
            result += str(diff_i) # This assumes C_root_i = 0 for free roots

    print(result)

# Using sys.stdin.readline is faster
sys.setrecursionlimit(300000) # Increase recursion depth for DSU find


solve()