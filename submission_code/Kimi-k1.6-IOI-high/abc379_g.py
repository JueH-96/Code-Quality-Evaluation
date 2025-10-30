MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    grid = []
    for _ in range(H):
        grid.append(data[idx])
        idx += 1
    
    # Precompute allowed values for each cell
    allowed = []
    for i in range(H):
        row = []
        for j in range(W):
            c = grid[i][j]
            if c == '?':
                row.append({1, 2, 3})
            else:
                row.append({int(c)})
        allowed.append(row)
    
    if H == 0 or W == 0:
        print(0)
        return
    
    # Initialize previous_dp and previous_total for the first row
    prev_dp = []
    prev_total = []
    
    # First row
    j = 0
    dp_j0 = [0] * 4
    allowed_vals = allowed[0][j]
    for c in allowed_vals:
        dp_j0[c] = 1
    prev_dp.append(dp_j0)
    prev_total.append(sum(dp_j0[1:4]) % MOD)
    
    for j in range(1, W):
        current_dp = [0] * 4
        allowed_vals = allowed[0][j]
        total_prev = prev_total[j-1]
        prev_j_minus_1_dp = prev_dp[j-1]
        for c in allowed_vals:
            sum_left_not_c = (total_prev - prev_j_minus_1_dp[c]) % MOD
            current_dp[c] = sum_left_not_c
        current_total = sum(current_dp[1:4]) % MOD
        prev_dp.append(current_dp)
        prev_total.append(current_total)
    
    # Process subsequent rows
    for i in range(1, H):
        current_dp = []
        current_total = []
        # Process j=0
        allowed_vals_j0 = allowed[i][0]
        new_j0_dp = [0] * 4
        prev_total_j0 = prev_total[0] if prev_total else 0
        prev_j0_dp = prev_dp[0] if prev_dp else [0] * 4
        for c in allowed_vals_j0:
            sum_prev_not_c = (prev_total_j0 - prev_j0_dp[c]) % MOD
            new_j0_dp[c] = sum_prev_not_c
        current_total_j0 = sum(new_j0_dp[1:4]) % MOD
        current_dp.append(new_j0_dp)
        current_total.append(current_total_j0)
        
        # Process j > 0
        for j in range(1, W):
            allowed_vals = allowed[i][j]
            new_dp = [0] * 4
            prev_j_dp = prev_dp[j] if j < len(prev_dp) else [0]*4
            prev_j_total = prev_total[j] if j < len(prev_total) else 0
            left_j_minus_1_dp = current_dp[j-1] if j-1 < len(current_dp) else [0]*4
            left_j_minus_1_total = current_total[j-1] if j-1 < len(current_total) else 0
            
            for c in allowed_vals:
                sum_prev_not_c = (prev_j_total - prev_j_dp[c]) % MOD
                sum_left_not_c = (left_j_minus_1_total - left_j_minus_1_dp[c]) % MOD
                new_dp[c] = (sum_prev_not_c * sum_left_not_c) % MOD
            
            current_total_j = sum(new_dp[1:4]) % MOD
            current_dp.append(new_dp)
            current_total.append(current_total_j)
        
        prev_dp = current_dp
        prev_total = current_total
    
    print(prev_total[-1] % MOD if prev_total else 0)

if __name__ == "__main__":
    main()