import sys

def mat_mul(A_mat, B_mat, N_val, mod_val):
    C_mat = [[0] * N_val for _ in range(N_val)]
    for i in range(N_val):
        for j in range(N_val):
            sum_val = 0
            for k_val in range(N_val):
                sum_val = (sum_val + A_mat[i][k_val] * B_mat[k_val][j])
            C_mat[i][j] = sum_val % mod_val # Modulo at the end of sum for efficiency
    return C_mat

def mat_pow(A_mat, exp_val, N_val, mod_val):
    res_mat = [[0] * N_val for _ in range(N_val)]
    for i in range(N_val):
        res_mat[i][i] = 1
    
    base_mat = A_mat
    while exp_val > 0:
        if exp_val % 2 == 1:
            res_mat = mat_mul(res_mat, base_mat, N_val, mod_val)
        base_mat = mat_mul(base_mat, base_mat, N_val, mod_val)
        exp_val //= 2
    return res_mat

def solve():
    N, p_mod = map(int, sys.stdin.readline().split())
    A_orig_mat = []
    for _ in range(N):
        A_orig_mat.append(list(map(int, sys.stdin.readline().split())))

    # A0 is A_orig_mat itself, as its zero entries are already 0.
    A0_mat = A_orig_mat 

    zeros_count = 0
    zero_locs_list = []
    for r_idx in range(N):
        for c_idx in range(N):
            if A_orig_mat[r_idx][c_idx] == 0:
                zeros_count += 1
                zero_locs_list.append((r_idx, c_idx))

    if p_mod == 2:
        A1_mat = [[A_orig_mat[i][j] for j in range(N)] for i in range(N)]
        for r_zero, c_zero in zero_locs_list:
            A1_mat[r_zero][c_zero] = 1
        
        # For p=2, the sum is (A1_mat)^p mod p. Here p=2.
        ans_matrix = mat_pow(A1_mat, p_mod, N, p_mod)
        
        for i in range(N):
            print(*(ans_matrix[i]))
        return

    # Case p_mod > 2
    ans_matrix = mat_pow(A0_mat, p_mod, N, p_mod)
    
    for r_k, c_k in zero_locs_list:
        if r_k == c_k: # Diagonal zero at (r_k, r_k)
            # Add (A0 * E_rkrk) term: add r_k-th col of A0 to r_k-th col of ans_matrix
            for i in range(N):
                ans_matrix[i][r_k] = (ans_matrix[i][r_k] + A0_mat[i][r_k]) % p_mod
            # Add (E_rkrk * A0) term: add r_k-th row of A0 to r_k-th row of ans_matrix
            for j in range(N):
                ans_matrix[r_k][j] = (ans_matrix[r_k][j] + A0_mat[r_k][j]) % p_mod
        else: # Off-diagonal zero at (r_k, c_k)
            if p_mod == 3:
                # Add (E_rkck * A0 * E_rkck) = A0[c_k][r_k] * E_rkck
                val_to_add = A0_mat[c_k][r_k]
                ans_matrix[r_k][c_k] = (ans_matrix[r_k][c_k] + val_to_add) % p_mod
    
    # Multiply by (p-1)^K = (-1)^K mod p_mod
    if zeros_count > 0 and zeros_count % 2 == 1: # If K is odd
        for r_idx in range(N):
            for c_idx in range(N):
                ans_matrix[r_idx][c_idx] = (p_mod - ans_matrix[r_idx][c_idx]) % p_mod
    
    for i in range(N):
        print(*(ans_matrix[i]))

solve()