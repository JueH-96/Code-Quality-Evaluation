import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S_grid = [sys.stdin.readline().strip() for _ in range(H)]

    MOD = 998244353

    # Check row conditions: count of 'A's in each row must be even
    possible_H = True
    for r in range(H):
        count_A_in_row = 0
        for c in range(W):
            if S_grid[r][c] == 'A':
                count_A_in_row += 1
        if count_A_in_row % 2 != 0:
            possible_H = False
            break
    
    if not possible_H:
        sys.stdout.write("0
")
        return

    # Check column conditions: count of 'A's in each column must be even
    possible_V = True
    for c in range(W):
        count_A_in_col = 0
        for r in range(H):
            if S_grid[r][c] == 'A':
                count_A_in_col += 1
        if count_A_in_col % 2 != 0:
            possible_V = False
            break
            
    if not possible_V:
        sys.stdout.write("0
")
        return
        
    # If all row and column parity conditions are met,
    # there are 2^H ways for horizontal choices and 2^W for vertical.
    # Total ways = 2^H * 2^W = 2^(H+W).
    ans = pow(2, H + W, MOD)
    sys.stdout.write(str(ans) + "
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()