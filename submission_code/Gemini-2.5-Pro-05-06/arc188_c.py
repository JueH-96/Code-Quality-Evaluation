import sys

sys.setrecursionlimit(4 * 10**5) # Increased recursion limit for DSU
input = sys.stdin.readline

def solve():
    N, M = map(int, input().split())
    testimonies = []
    for _ in range(M):
        u, v, c = map(int, input().split())
        testimonies.append((u - 1, v - 1, c)) # 0-indexed villagers

    parent = list(range(N))
    # val[i] stores X_i ^ X_parent[i] initially.
    # After find(i) is called, parent[i] becomes the root,
    # and val[i] stores X_i ^ X_root[i].
    val = [0] * N 
    # Y[i] stores confusion status of villager i. -1 means unset.
    Y_status = [-1] * N # Renamed to avoid conflict if N is used as a variable name elsewhere

    def find(i):
        if parent[i] == i:
            return i, 0 # (root, X_i ^ X_root)
        
        # Recursively find root of parent and X_parent[i] ^ X_root
        root, d_parent_root = find(parent[i])
        
        # Path compression: make i a direct child of root
        parent[i] = root
        
        # Update val[i]:
        # Original val[i] was X_i ^ X_original_parent[i]
        # d_parent_root is X_original_parent[i] ^ X_root
        # New val[i] becomes (X_i ^ X_original_parent[i]) ^ (X_original_parent[i] ^ X_root)
        # which simplifies to X_i ^ X_root
        val[i] = val[i] ^ d_parent_root
        return root, val[i]

    # Union sets with roots r1, r2. K_roots = X_r1 ^ X_r2.
    # Make r2 a child of r1.
    def union_sets(r1, r2, K_roots):
        parent[r2] = r1
        # val[r2] must store X_r2 ^ X_parent[r2] = X_r2 ^ X_r1
        # Since K_roots = X_r1 ^ X_r2, then X_r2 ^ X_r1 = K_roots.
        val[r2] = K_roots


    for i in range(M):
        Ai, Bi, Ci = testimonies[i] # Ai is the testifier

        R_A, D_A = find(Ai) # D_A = X_Ai ^ X_R_A
        R_B, D_B = find(Bi) # D_B = X_Bi ^ X_R_B

        # Equation from problem: X_Ai ^ X_Bi ^ Y_Ai = Ci
        # Substitute known relations: (D_A ^ X_R_A) ^ (D_B ^ X_R_B) ^ Y_Ai = Ci
        # Rearrange: D_A ^ D_B ^ X_R_A ^ X_R_B ^ Y_Ai = Ci

        if R_A == R_B:
            # Components are already connected. X_R_A ^ X_R_B = 0.
            # Equation becomes: D_A ^ D_B ^ Y_Ai = Ci
            # This determines Y_Ai: Y_Ai = D_A ^ D_B ^ Ci
            required_Y_Ai = D_A ^ D_B ^ Ci
            
            if Y_status[Ai] == -1: # Y_Ai was not set
                Y_status[Ai] = required_Y_Ai
            elif Y_status[Ai] != required_Y_Ai: # Y_Ai was set but conflicts
                print("-1")
                return
        else:
            # Components are different. They need to be merged.
            # Equation: X_R_A ^ X_R_B = D_A ^ D_B ^ Ci ^ Y_Ai
            
            # Determine Y_Ai value to use for this merge
            y_ai_to_use = Y_status[Ai]
            if y_ai_to_use == -1: # Y_Ai not set, greedily set to 0
                y_ai_to_use = 0
                Y_status[Ai] = 0
            
            # K_roots is the required XOR sum between X_R_A and X_R_B
            K_roots = D_A ^ D_B ^ Ci ^ y_ai_to_use
            union_sets(R_A, R_B, K_roots)

    # Finalize Y_status: any -1 becomes 0
    ans_str_list = []
    for i in range(N):
        if Y_status[i] == -1:
            ans_str_list.append('0')
        else:
            ans_str_list.append(str(Y_status[i]))
            
    print("".join(ans_str_list))

solve()