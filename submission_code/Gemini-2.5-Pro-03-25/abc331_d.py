# YOUR CODE HERE
import sys

def solve():
    # Use sys.stdin.readline for faster input reading, especially important for large Q
    input = sys.stdin.readline
    
    # Read N (pattern size) and Q (number of queries)
    N, Q = map(int, input().split())
    
    # Read the N x N pattern P
    P = [input().strip() for _ in range(N)]
    
    # Build a 2D prefix sum table `ps` for the pattern P.
    # ps[i][j] will store the count of black squares ('B') in the rectangle
    # defined by top-left corner (0, 0) and bottom-right corner (i-1, j-1) within the pattern P.
    # The table size is (N+1) x (N+1) to handle boundary conditions easily (0-based indices i, j map to ps[i+1][j+1]).
    ps = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            # The prefix sum recurrence relation:
            # ps[i+1][j+1] = value_at(i,j) + ps[i][j+1] (sum above) + ps[i+1][j] (sum left) - ps[i][j] (sum top-left, subtracted twice)
            # value_at(i,j) is 1 if P[i][j] is 'B', else 0.
            ps[i+1][j+1] = ps[i][j+1] + ps[i+1][j] - ps[i][j] + (1 if P[i][j] == 'B' else 0)

    # Helper function to calculate the number of black squares within a given rectangular subgrid of the N x N pattern P.
    # The rectangle is defined by top-left corner (r1, c1) and bottom-right corner (r2, c2), inclusive. Indices are 0-based.
    def count_pattern(r1, c1, r2, c2):
        # If the rectangle is invalid (e.g., r1 > r2 or c1 > c2), it contains 0 black squares.
        # This handles cases where remaining rows/columns (r_r, r_c) are 0, resulting in indices like -1.
        if r1 > r2 or c1 > c2:
            return 0
        
        # Calculate the count using the precomputed prefix sum table `ps`.
        # The count for rectangle (r1, c1) to (r2, c2) is obtained via inclusion-exclusion on prefix sums:
        # Total count up to (r2, c2) = ps[r2+1][c2+1]
        # Subtract count up to (r1-1, c2) = ps[r1][c2+1]
        # Subtract count up to (r2, c1-1) = ps[r2+1][c1]
        # Add back count up to (r1-1, c1-1) = ps[r1][c1] (subtracted twice)
        res = ps[r2+1][c2+1] - ps[r1][c2+1] - ps[r2+1][c1] + ps[r1][c1]
        return res

    # Calculate the total number of black squares in one full N x N pattern block.
    # This is simply the value at ps[N][N], which covers the rectangle (0,0) to (N-1, N-1).
    total_black_in_N_by_N = ps[N][N] 

    # Main function to count black squares in the large grid, specifically in the rectangle
    # defined by top-left corner (0, 0) and bottom-right corner (R, S).
    def count_black(R, S):
        # If R or S is negative, the rectangle is empty or outside the valid grid area (0,0 onwards). Return 0.
        if R < 0 or S < 0:
            return 0

        # Calculate the dimensions of the rectangle. Since indices are inclusive,
        # the rectangle from (0,0) to (R,S) has R+1 rows and S+1 columns.
        num_rows = R + 1
        num_cols = S + 1
        
        # Determine how many full N x N blocks fit into this rectangle, and the dimensions of the remaining partial block.
        # q_r: number of full N-row blocks vertically.
        # r_r: number of remaining rows in the partial block at the bottom (0 <= r_r < N).
        q_r = num_rows // N  
        r_r = num_rows % N   
        
        # q_c: number of full N-column blocks horizontally.
        # r_c: number of remaining columns in the partial block to the right (0 <= r_c < N).
        q_c = num_cols // N  
        r_c = num_cols % N   

        # The total count is the sum of black squares from four parts based on grid tiling:
        # 1. The area covered by q_r * q_c full N x N blocks.
        res = q_r * q_c * total_black_in_N_by_N
        
        # 2. The area consisting of q_r full N-row blocks stacked vertically, each covering columns 0 to r_c-1.
        # This contributes q_r times the count of black squares in the pattern P for rectangle (0, 0) to (N-1, r_c-1).
        res += q_r * count_pattern(0, 0, N-1, r_c - 1) 
        
        # 3. The area consisting of q_c full N-column blocks placed horizontally, each covering rows 0 to r_r-1.
        # This contributes q_c times the count of black squares in the pattern P for rectangle (0, 0) to (r_r-1, N-1).
        res += q_c * count_pattern(0, 0, r_r - 1, N-1)
        
        # 4. The small rectangular area at the bottom-right corner, with dimensions r_r x r_c.
        # This corresponds to one partial pattern block covering rectangle (0, 0) to (r_r-1, r_c-1).
        res += count_pattern(0, 0, r_r - 1, r_c - 1)
        
        return res

    # Process each of the Q queries
    for _ in range(Q):
        # Read the query coordinates A, B, C, D
        A, B, C, D = map(int, input().split())
        
        # Apply the principle of inclusion-exclusion using the count_black function.
        # The number of black squares in the rectangle defined by top-left (A, B) and bottom-right (C, D) is:
        # Count((0,0) to (C,D)) - Count((0,0) to (A-1, D)) - Count((0,0) to (C, B-1)) + Count((0,0) to (A-1, B-1))
        # The count_black function handles negative coordinates by returning 0, which correctly deals with boundary cases (A=0 or B=0).
        ans = count_black(C, D) - count_black(A - 1, D) - count_black(C, B - 1) + count_black(A - 1, B - 1)
        
        # Print the result for the current query
        print(ans)

# Call the main function to execute the solution logic
solve()