import sys
from collections import defaultdict

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())
    
    # Precompute allowed colors for each cell
    allowed = [[[] for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == '?':
                allowed[i][j] = [1, 2, 3]
            else:
                allowed[i][j] = [int(c)]
    
    # Initialize DP for the first row
    prev_dp = [{} for _ in range(W)]
    for j in range(W):
        if j == 0:
            colors = allowed[0][0]
            for c in colors:
                prev_dp[j][c] = 1
        else:
            prev_col = prev_dp[j-1]
            current_allowed = allowed[0][j]
            current_dict = {}
            for prev_color in prev_col:
                for current_color in current_allowed:
                    if current_color != prev_color:
                        current_dict[current_color] = current_dict.get(current_color, 0) + prev_col[prev_color]
            # Apply modulo
            for k in current_dict:
                current_dict[k] %= MOD
            prev_dp[j] = current_dict
    
    # Process remaining rows
    for i in range(1, H):
        current_dp = [{} for _ in range(W)]
        for j in range(W):
            if j == 0:
                current_allowed = allowed[i][0]
                top_colors = prev_dp[0]
                current_dict = {}
                for top_color in top_colors:
                    for current_color in current_allowed:
                        if current_color != top_color:
                            cnt = top_colors[top_color]
                            current_dict[current_color] = current_dict.get(current_color, 0) + cnt
                current_dp[j] = current_dict
            else:
                current_allowed = allowed[i][j]
                top_colors = prev_dp[j]
                prev_col_dict = current_dp[j-1]
                current_dict = {}
                for top_color in top_colors:
                    for prev_color in prev_col_dict:
                        for current_color in current_allowed:
                            if current_color != prev_color and current_color != top_color:
                                cnt = top_colors[top_color] * prev_col_dict[prev_color]
                                current_dict[current_color] = current_dict.get(current_color, 0) + cnt
                # Apply modulo
                for k in current_dict:
                    current_dict[k] %= MOD
                current_dp[j] = current_dict
        prev_dp = current_dp
    
    # The answer is the sum of the last column's counts
    if not prev_dp[-1]:
        print(0)
    else:
        ans = sum(prev_dp[-1].values()) % MOD
        print(ans)

if __name__ == "__main__":
    main()