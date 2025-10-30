import sys

def solve():
    H_orig, W_orig = map(int, sys.stdin.readline().split())
    
    # Use H, W for 0-indexed dimensions
    H, W = H_orig, W_orig

    A_grid = []
    for _ in range(H_orig):
        A_grid.append(list(map(int, sys.stdin.readline().split())))

    Q_count, sh_start, sw_start = map(int, sys.stdin.readline().split())
    sh_curr = sh_start - 1 # 0-indexed
    sw_curr = sw_start - 1 # 0-indexed

    queries_data = []
    for _ in range(Q_count):
        line = sys.stdin.readline().split()
        d_i = line[0]
        a_i = int(line[1])
        queries_data.append((d_i, a_i))

    MOD = 998244353

    dp_from_start = [[0] * W for _ in range(H)]
    dp_to_end = [[0] * W for _ in range(H)]

    def compute_dp_from_start_region(r_min, c_min):
        for r in range(r_min, H):
            for c in range(c_min, W):
                val_A = A_grid[r][c]
                sum_prev = 0
                if r == 0 and c == 0:
                    sum_prev = 1 
                else:
                    if r > 0:
                        sum_prev = (sum_prev + dp_from_start[r-1][c]) % MOD
                    if c > 0:
                        sum_prev = (sum_prev + dp_from_start[r][c-1]) % MOD
                dp_from_start[r][c] = (sum_prev * val_A) % MOD
    
    def compute_dp_to_end_region(r_max, c_max):
        for r in range(r_max, -1, -1):
            for c in range(c_max, -1, -1):
                val_A = A_grid[r][c]
                sum_next = 0
                if r == H-1 and c == W-1:
                    sum_next = 1
                else:
                    if r < H-1:
                        sum_next = (sum_next + dp_to_end[r+1][c]) % MOD
                    if c < W-1:
                        sum_next = (sum_next + dp_to_end[r][c+1]) % MOD
                dp_to_end[r][c] = (sum_next * val_A) % MOD

    # Initial full computation
    compute_dp_from_start_region(0, 0)
    compute_dp_to_end_region(H-1, W-1)

    results = []

    for d_char, target_a_val in queries_data:
        # Update current position based on d_char
        if d_char == 'L':
            sw_curr -= 1
        elif d_char == 'R':
            sw_curr += 1
        elif d_char == 'U':
            sh_curr -= 1
        elif d_char == 'D':
            sh_curr += 1
        
        r_changed, c_changed = sh_curr, sw_curr
        
        old_A_rc_val = A_grid[r_changed][c_changed]

        # Calculate L_val
        L_val = 0
        if r_changed == 0 and c_changed == 0:
            L_val = 1
        else:
            if r_changed > 0:
                L_val = (L_val + dp_from_start[r_changed-1][c_changed]) % MOD
            if c_changed > 0:
                L_val = (L_val + dp_from_start[r_changed][c_changed-1]) % MOD
        
        # Calculate R_val
        R_val = 0
        if r_changed == H-1 and c_changed == W-1:
            R_val = 1
        else:
            if r_changed < H-1:
                R_val = (R_val + dp_to_end[r_changed+1][c_changed]) % MOD
            if c_changed < W-1:
                R_val = (R_val + dp_to_end[r_changed][c_changed+1]) % MOD
        
        # Current total sum from (0,0) to (H-1,W-1)
        # This value is from the DP table *before* reflecting the change at (r_changed, c_changed)
        # This is S_old
        s_old = dp_from_start[H-1][W-1] 
        
        # Calculate change in sum
        # (A_new - A_old) * L_val * R_val
        term_coeff = (L_val * R_val) % MOD
        val_diff = (target_a_val - old_A_rc_val + MOD) % MOD 
        delta_s = (val_diff * term_coeff) % MOD
        
        s_new = (s_old + delta_s) % MOD
        results.append(s_new)
        
        # Update grid value
        A_grid[r_changed][c_changed] = target_a_val
        
        # Update DP tables to reflect the change for future queries
        compute_dp_from_start_region(r_changed, c_changed)
        compute_dp_to_end_region(r_changed, c_changed)
        # Note: after this, dp_from_start[H-1][W-1] will be s_new

    sys.stdout.write('
'.join(map(str, results)) + '
')

solve()