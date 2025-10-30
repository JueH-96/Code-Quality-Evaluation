import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    min_r_hash = H 
    max_r_hash = -1
    min_c_hash = W
    max_c_hash = -1

    # Find the bounding box of existing '#' cells.
    # The problem guarantees at least one '#' cell exists.
    for r_idx in range(H):
        for c_idx in range(W):
            if S[r_idx][c_idx] == '#':
                min_r_hash = min(min_r_hash, r_idx)
                max_r_hash = max(max_r_hash, r_idx)
                min_c_hash = min(min_c_hash, c_idx)
                max_c_hash = max(max_c_hash, c_idx)
    
    # Let B_hash be the minimal bounding box of initial '#' cells.
    # B_hash is defined by rows [min_r_hash, max_r_hash] 
    # and columns [min_c_hash, max_c_hash].
    #
    # A necessary condition: B_hash must not contain any '.' cells.
    # (Proof sketch: If B_hash contains a '.', say at (r0, c0), then
    # since any valid final black rectangle R_final must contain B_hash,
    # R_final would also contain (r0, c0). But initial '.' cells must be
    # outside R_final, leading to a contradiction.)
    
    # Check this necessary condition:
    for r_idx in range(min_r_hash, max_r_hash + 1):
        for c_idx in range(min_c_hash, max_c_hash + 1):
            if S[r_idx][c_idx] == '.':
                print("No")
                return
    
    # This condition is also sufficient. If B_hash contains no '.' cells,
    # we can choose the final black rectangle R_final to be exactly B_hash.
    # 1. All cells (i,j) within B_hash are ultimately black.
    #    - If S[i][j] was '#', it remains black. (Consistent)
    #    - If S[i][j] was '.', this case is ruled out by the check above.
    #    - If S[i][j] was '?', it's painted black. (Consistent)
    # 2. All cells (i,j) outside B_hash are ultimately white.
    #    - If S[i][j] was '#', this contradicts the definition of B_hash
    #      (as B_hash is the *minimal* box containing *all* initial '#').
    #      So this case cannot happen.
    #    - If S[i][j] was '.', it remains white. (Consistent)
    #    - If S[i][j] was '?', it's painted white. (Consistent)
    # Thus, if the check passes, "Yes" is the answer.
    
    print("Yes")

solve()