import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())

    # Step 1: Populate N[P], counts of each row pattern P
    counts_N = [0] * (1 << W)
    for _ in range(H):
        s = sys.stdin.readline().strip()
        mask = 0
        for j in range(W): # s[j] is j-th char from left (0-indexed)
            if s[j] == '1':
                # s[j] corresponds to j-th bit (0 is LSB)
                # e.g. W=3, "100" -> s[0]='1', s[1]='0', s[2]='0'
                # mask = (1<<0) = 1, i.e. 001_2
                mask |= (1 << j) 
        counts_N[mask] += 1

    # Step 2: Populate F[M]
    F = [0] * (1 << W)
    for mask_val in range(1 << W):
        popcount = bin(mask_val).count('1') 
        F[mask_val] = min(popcount, W - popcount)

    # FWHT function (defined as nested or outside)
    # P is modified in-place
    def do_fwht(P_arr, is_inverse):
        n_len = len(P_arr)
        h = 1
        while h < n_len:
            for i_block_start in range(0, n_len, h * 2):
                for j_idx in range(i_block_start, i_block_start + h):
                    x = P_arr[j_idx]
                    y = P_arr[j_idx + h]
                    P_arr[j_idx] = x + y
                    P_arr[j_idx + h] = x - y
            h *= 2
        
        if is_inverse:
            for i_idx in range(n_len):
                P_arr[i_idx] //= n_len
    
    # Step 3: Transform N and F
    # Make copies as FWHT is in-place
    N_hat = list(counts_N) 
    F_hat = list(F)        

    do_fwht(N_hat, False)
    do_fwht(F_hat, False)

    # Step 4: Element-wise product
    S_hat = [0] * (1 << W)
    for i in range(1 << W):
        S_hat[i] = N_hat[i] * F_hat[i]

    # Step 5: Inverse transform S_hat
    do_fwht(S_hat, True) # S_hat now becomes S

    # Step 6: Find minimum in S
    min_total_sum = H * W # Initialize with a value >= max possible sum
    for val in S_hat:
        if val < min_total_sum:
            min_total_sum = val
            
    sys.stdout.write(str(min_total_sum) + "
")

solve()