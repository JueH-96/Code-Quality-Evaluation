# YOUR CODE HERE
N, M = map(int, input().split())

S_grid = []
for _ in range(N):
    S_grid.append(input())

T_grid = []
for _ in range(M):
    T_grid.append(input())

# Iterate through all possible top-left starting positions (0-based) in S_grid
# r_s_0based can go from 0 to N-M
# c_s_0based can go from 0 to N-M
for r_s_0based in range(N - M + 1):
    for c_s_0based in range(N - M + 1):
        
        current_subgrid_matches_T = True # Flag to track if the current subgrid matches T
        
        # Compare the M x M subgrid of S (starting at r_s_0based, c_s_0based) with T_grid
        for i_t_0based in range(M):  # 0-based row index in T
            for j_t_0based in range(M):  # 0-based column index in T
                
                # Character in S_grid: S_grid[r_s_0based + i_t_0based][c_s_0based + j_t_0based]
                # Character in T_grid: T_grid[i_t_0based][j_t_0based]
                if S_grid[r_s_0based + i_t_0based][c_s_0based + j_t_0based] != T_grid[i_t_0based][j_t_0based]:
                    current_subgrid_matches_T = False
                    break  # Mismatch found, break from inner j_t_0based loop
            
            if not current_subgrid_matches_T:
                break  # Mismatch found, break from i_t_0based loop
        
        if current_subgrid_matches_T:
            # If flag is still True, a match is found
            # Convert 0-based r_s_0based, c_s_0based to 1-based a, b for output
            a_1based = r_s_0based + 1
            b_1based = c_s_0based + 1
            
            print(a_1based, b_1based)
            
            # Problem states there is exactly one pair (a,b), so we can exit after finding it.
            exit()