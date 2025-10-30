import sys
from collections import deque

def main():
    """
    Solves the problem by first labeling initial green components, then calculating
    the expected number of components after changing one red cell to green.
    """
    # It's good practice to increase the recursion limit for deep DFS,
    # though we use an iterative BFS approach here.
    sys.setrecursionlimit(2 * 10**6)

    MOD = 998244353

    try:
        H, W = map(int, sys.stdin.readline().split())
        S = [sys.stdin.readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handle cases with empty or malformed input.
        return

    # --- Step 1: Component Labeling of initial green cells ---
    # We use a BFS-based approach to find and label all connected components.
    
    comp_id = [[0] * W for _ in range(H)]
    n0 = 0  # Initial number of green components
    
    for r in range(H):
        for c in range(W):
            if S[r][c] == '#' and comp_id[r][c] == 0:
                n0 += 1
                q = deque([(r, c)])
                comp_id[r][c] = n0
                
                while q:
                    curr_r, curr_c = q.popleft()
                    
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        if 0 <= nr < H and 0 <= nc < W and S[nr][nc] == '#' and comp_id[nr][nc] == 0:
                            comp_id[nr][nc] = n0
                            q.append((nr, nc))

    # --- Step 2: Calculate sum_k and count red cells ---
    # For each red cell, we determine `k`, the number of distinct green components
    # it is adjacent to. We sum these `k` values over all red cells.
    
    num_red = 0
    sum_k = 0
    
    for r in range(H):
        for c in range(W):
            if S[r][c] == '.':
                num_red += 1
                
                neighbor_comps = set()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and S[nr][nc] == '#':
                        neighbor_comps.add(comp_id[nr][nc])
                
                k = len(neighbor_comps)
                sum_k += k

    # --- Step 3: Final Calculation ---
    # The expected value E can be expressed as:
    # E = (1 / num_red) * sum_{r in red_cells} (n0 + 1 - k(r))
    #   = (1 / num_red) * (num_red * (n0 + 1) - sum_k)
    # We calculate this value modulo MOD.

    # num_red is guaranteed to be at least 1 by the problem constraints.
    
    # Calculate the sum of components over all possibilities.
    # sum C(r) = |R|*(N0+1) - sum_k
    
    term_n0 = (num_red % MOD * n0 % MOD) % MOD
    term_num_red = num_red % MOD
    term_sum_k = sum_k % MOD

    # Use `+ MOD` to handle potential negative results before the final modulo.
    total_sum_of_components = (term_n0 + term_num_red - term_sum_k + MOD) % MOD
    
    # Modular multiplicative inverse of num_red.
    inv_num_red = pow(num_red, MOD - 2, MOD)
    
    answer = (total_sum_of_components * inv_num_red) % MOD
    print(answer)

if __name__ == "__main__":
    main()