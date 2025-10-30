def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    # read grid dimensions and grid
    H, W = map(int, input().split())
    A = [ list(map(int, input().split())) for _ in range(H) ]
    
    # dp[i][j] will store the dp value for cell (i,j) (0-indexed)
    dp = [[0]*W for _ in range(H)]
    dp[0][0] = A[0][0] % MOD
    for j in range(1,W):
        dp[0][j] = (A[0][j] % MOD * dp[0][j-1]) % MOD
    for i in range(1,H):
        dp[i][0] = (A[i][0] % MOD * dp[i-1][0]) % MOD
        for j in range(1,W):
            s = (dp[i-1][j] + dp[i][j-1]) % MOD
            dp[i][j] = (A[i][j] % MOD * s) % MOD

    # read Q and Takahashi's starting cell (1-indexed)
    Q, sh, sw = map(int, input().split())
    cur_r, cur_c = sh-1, sw-1  # current position (0-indexed)
    
    # For each update we propagate changes in a dependency region.
    # We use a row‐by‐row "affected" mask: in a given row, a cell is "affected"
    # if one of its dependents (top or left) was updated.
    out_lines = []
    for _ in range(Q):
        line = input().split()
        if not line: break
        d, new_val = line[0], int(line[1])
        # update Takahashi's current cell according to the move direction
        if d == 'L':
            cur_c -= 1
        elif d == 'R':
            cur_c += 1
        elif d == 'U':
            cur_r -= 1
        elif d == 'D':
            cur_r += 1
        
        ur, uc = cur_r, cur_c  # cell being updated
        # if the new value is same, no need to re-propagate:
        if A[ur][uc] == new_val:
            out_lines.append(str(dp[H-1][W-1] % MOD))
            continue
        A[ur][uc] = new_val

        # --- Propagate updates ---
        # For row 'ur': all columns from uc on are definitely affected.
        affected = [False]*W
        for j in range(uc, W):
            affected[j] = True
        # update row ur
        for j in range(uc, W):
            top = dp[ur-1][j] if ur > 0 else 0
            left = dp[ur][j-1] if j > 0 else 0
            # special-case: cell (0,0) already computed above
            if ur == 0 and j == 0:
                dp[0][0] = A[0][0] % MOD
            else:
                dp[ur][j] = (A[ur][j] * ((top + left) % MOD)) % MOD
        
        prev_aff = affected[:]   # affected mask for row ur
        # For subsequent rows, determine “affected” cells by:
        #   a cell in row i is affected if (i-1,j) was affected OR (i,j-1) was affected.
        for i in range(ur+1, H):
            curr_aff = [False]*W
            for j in range(W):
                if i-1 >= 0 and prev_aff[j]:
                    curr_aff[j] = True
                if j > 0 and curr_aff[j-1]:
                    curr_aff[j] = True
            # If no cell is affected in this row, break the propagation early.
            if not any(curr_aff):
                break
            for j in range(W):
                if curr_aff[j]:
                    top = dp[i-1][j] if i > 0 else 0
                    left = dp[i][j-1] if j > 0 else 0
                    dp[i][j] = (A[i][j] * ((top + left) % MOD)) % MOD
            prev_aff = curr_aff
        out_lines.append(str(dp[H-1][W-1] % MOD))
    sys.stdout.write("
".join(out_lines))
 
if __name__ == '__main__':
    main()