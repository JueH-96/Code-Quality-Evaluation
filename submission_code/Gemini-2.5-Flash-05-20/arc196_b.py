import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    ans = 0
    MOD = 998244353

    # Check if a configuration where all segments are horizontal is possible (all 'B' tiles)
    # This corresponds to H_config[i][j]=1, V_config[i][j]=0 for all i,j.
    # Each cell is in LR orientation (Type B).
    possible_all_horizontal_B = True
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'A': # 'A' tile cannot be LR type
                possible_all_horizontal_B = False
                break
        if not possible_all_horizontal_B:
            break
    if possible_all_horizontal_B:
        ans = (ans + 1) % MOD

    # Check if a configuration where all segments are vertical is possible (all 'B' tiles)
    # This corresponds to H_config[i][j]=0, V_config[i][j]=1 for all i,j.
    # Each cell is in TB orientation (Type B).
    possible_all_vertical_B = True
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'A': # 'A' tile cannot be TB type
                possible_all_vertical_B = False
                break
        if not possible_all_vertical_B:
            break
    if possible_all_vertical_B:
        ans = (ans + 1) % MOD

    # Check if a configuration where all segments form a checkerboard of 'A' tiles is possible (Pattern 3)
    # This corresponds to H_config[i][j] = (i+j)%2, V_config[i][j] = (i+j+1)%2.
    # This leads to BL/TR orientations. All cells must be Type A.
    possible_all_A_checkerboard_1 = True
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'B': # 'B' tile cannot be BL/TR type
                possible_all_A_checkerboard_1 = False
                break
        if not possible_all_A_checkerboard_1:
            break
    if possible_all_A_checkerboard_1:
        ans = (ans + 1) % MOD

    # Check if another configuration for all 'A' tiles is possible (Pattern 4)
    # This corresponds to H_config[i][j] = (i+j+1)%2, V_config[i][j] = (i+j)%2.
    # This leads to LT/RB orientations. All cells must be Type A.
    possible_all_A_checkerboard_2 = True
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'B': # 'B' tile cannot be LT/RB type
                possible_all_A_checkerboard_2 = False
                break
        if not possible_all_A_checkerboard_2:
            break
    if possible_all_A_checkerboard_2:
        ans = (ans + 1) % MOD

    # Special case: H and W are both odd.
    # There are two additional types of solutions for mixed A/B patterns.
    # These solutions arise from more complex configurations.
    if H % 2 == 1 and W % 2 == 1:
        # Pattern 5: S[i][j] is 'A' if (i+j) is even, 'B' if (i+j) is odd.
        matches_pattern_5_S = True
        for r in range(H):
            for c in range(W):
                expected_type = 'A' if (r + c) % 2 == 0 else 'B'
                if S[r][c] != expected_type:
                    matches_pattern_5_S = False
                    break
            if not matches_pattern_5_S:
                break
        if matches_pattern_5_S:
            ans = (ans + 2) % MOD # These specific patterns usually come in pairs of solutions.

        # Pattern 6: S[i][j] is 'A' if (i+j) is odd, 'B' if (i+j) is even.
        matches_pattern_6_S = True
        for r in range(H):
            for c in range(W):
                expected_type = 'A' if (r + c) % 2 == 1 else 'B'
                if S[r][c] != expected_type:
                    matches_pattern_6_S = False
                    break
            if not matches_pattern_6_S:
                break
        if matches_pattern_6_S:
            ans = (ans + 2) % MOD

    print(ans)


T = int(sys.stdin.readline())
for _ in range(T):
    solve()