import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    queries = [int(sys.stdin.readline()) for _ in range(Q)]

    # dp[r_used][c_used] will store a set of possible sums X = sum(r_p * c_p)
    # where r_used is the total number of rows accounted for across all 'mixed' SCCs
    # and c_used is the total number of columns accounted for across all 'mixed' SCCs.
    # 'Mixed' SCCs are those containing at least one row node and at least one column node.
    # The remaining (N - r_used) rows and (N - c_used) columns form 'pure' SCCs
    # (containing only row nodes or only column nodes) and contribute 0 to the sum X.
    
    # Initialize dp table. dp[0][0] contains 0, representing the starting state
    # where no rows or columns are assigned to mixed SCCs, thus X=0.
    # All other dp[i][j] are empty sets initially.
    dp = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    dp[0][0].add(0)

    # Iterate through all possible (r_used, c_used) states.
    # r_used and c_used represent the total rows/columns already assigned to mixed SCCs.
    for r_used in range(N + 1):
        for c_used in range(N + 1):
            if not dp[r_used][c_used]:
                continue # Skip if this state (r_used, c_used) is not reachable

            # For each possible sum 'current_X' achieved for the state (r_used, c_used)
            # Iterate over a copy of the set as it might be modified during inner loops.
            for current_X in list(dp[r_used][c_used]):
                # Try adding a new 'mixed' SCC with dimensions (r_new, c_new).
                # r_new and c_new must be at least 1, as they represent part of a 'mixed' SCC
                # and must fit within the remaining available rows/columns.
                for r_new in range(1, N - r_used + 1):
                    for c_new in range(1, N - c_used + 1):
                        new_r_used = r_used + r_new
                        new_c_used = c_used + c_new
                        new_X = current_X + (r_new * c_new)
                        
                        # Add new_X to the set for (new_r_used, new_c_used).
                        # The maximum possible value for X is N*N.
                        if new_X <= N * N:
                            dp[new_r_used][new_c_used].add(new_X)

    # After filling the DP table, dp[N][N] contains all possible values for X
    # (the total number of non-fixed elements) when all N rows and N columns
    # are accounted for (either by mixed SCCs or implicitly by pure SCCs).
    
    possible_fixed_K_values = set()
    
    # K = N^2 - X. For every X value in dp[N][N], calculate the corresponding K.
    for X_val in dp[N][N]:
        possible_fixed_K_values.add(N * N - X_val)

    # Special case: X=0 (no non-fixed elements) corresponds to K=N^2 (all elements fixed).
    # This scenario is explicitly added here, as dp[N][N] might not contain 0 if N > 0.
    # For instance, an all-zero matrix has N^2 fixed elements, resulting in X=0.
    possible_fixed_K_values.add(N * N) 

    # Process queries
    results = []
    for q_K in queries:
        if q_K in possible_fixed_K_values:
            results.append("Yes")
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")

solve()