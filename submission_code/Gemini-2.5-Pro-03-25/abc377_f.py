# YOUR CODE HERE
import sys

def solve():
    # Read N (grid size) and M (number of pieces) from standard input.
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize sets to store unique row indices, column indices,
    # sum diagonal indices (i+j), and difference diagonal indices (i-j)
    # that are "attacked" by the existing pieces.
    A_set = set() # Set of row indices with pieces
    B_set = set() # Set of column indices with pieces
    S_set = set() # Set of sum diagonal indices (i+j) corresponding to piece locations
    D_set = set() # Set of difference diagonal indices (i-j) corresponding to piece locations
    
    # Read the coordinates of the M pieces and populate the sets.
    # Each piece at (a, b) "captures" row 'a', column 'b', sum diagonal 'a+b', and difference diagonal 'a-b'.
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        A_set.add(a)
        B_set.add(b)
        S_set.add(a + b)
        D_set.add(a - b)

    # Calculate the number of distinct rows and columns that contain pieces.
    Mr = len(A_set) # Number of distinct rows with pieces
    Mc = len(B_set) # Number of distinct columns with pieces

    # Calculate the initial count of potentially safe squares.
    # This is the number of squares (i, j) such that row 'i' does not contain a piece AND column 'j' does not contain a piece.
    # This count represents |G'| = |{(i, j) | 1 <= i, j <= N, i not in A_set, j not in B_set}|
    # We start with this count and will subtract squares that are captured by diagonals.
    TotalSafe_initial = (N - Mr) * (N - Mc)

    # Calculate |U_sum|, the count of squares (i, j) such that i is not in A_set, j is not in B_set, but i+j is in S_set.
    # These are squares safe from row/column captures but captured by a sum diagonal.
    # We compute this using the Principle of Inclusion-Exclusion for each relevant sum diagonal 's'.
    U_sum_val = 0
    for s in S_set:
        # Determine the range of row indices 'i' for points (i, s-i) on the sum diagonal s that lie within the NxN grid.
        # A point (i, j) is on diagonal s if i+j = s. It's within the grid if 1 <= i <= N and 1 <= j <= N.
        # Substituting j = s-i gives 1 <= s-i <= N.
        # Combining 1 <= i <= N and 1 <= s-i <= N yields the valid range for i:
        min_i = max(1, s - N)
        max_i = min(N, s - 1)
        
        # If min_i > max_i, the diagonal segment defined by s does not contain any points within the grid. Skip this s.
        if min_i > max_i: 
             continue

        # Calculate the total number of integer points on this diagonal segment within the grid bounds.
        Ps_count = max_i - min_i + 1 

        # Calculate adjustments based on Inclusion-Exclusion Principle.
        # We need to count points on the segment (i, s-i) where i is in A_set OR j=s-i is in B_set.
        
        Cnt_A = 0 # Counts points (i, s-i) on the segment where the row index i is in A_set (captured row).
        for r_idx in A_set:
             # Check if this row index r_idx falls within the valid range [min_i, max_i] for the current diagonal segment.
             if min_i <= r_idx <= max_i:
                  Cnt_A += 1
        
        Cnt_B = 0 # Counts points (i, s-i) on the segment where the column index j=s-i is in B_set (captured column).
        for c_idx in B_set:
             # Calculate the corresponding row index i for column index j=c_idx on diagonal s.
             i = s - c_idx 
             # Check if this row index i is within the valid range [min_i, max_i] for the current diagonal segment.
             if min_i <= i <= max_i: 
                  Cnt_B += 1

        Cnt_AB = 0 # Counts points (i, s-i) on the segment where i is in A_set AND j=s-i is in B_set.
        for r_idx in A_set:
             # Check if row index i=r_idx is within the valid range...
             if min_i <= r_idx <= max_i:
                 # ...and if the corresponding column index j=s-r_idx is in B_set.
                 if (s - r_idx) in B_set:
                      Cnt_AB += 1
        
        # Apply Inclusion-Exclusion: The number of points on diagonal s that are safe from row/column captures is
        # Ps_count - (points captured by row A) - (points captured by column B) + (points captured by both).
        Cs = Ps_count - Cnt_A - Cnt_B + Cnt_AB
        # Accumulate this count for diagonal s into the total U_sum_val.
        U_sum_val += Cs

    # Calculate |U_diff|, the count of squares (i, j) such that i not in A_set, j not in B_set, but i-j in D_set.
    # These are squares safe from row/column captures but captured by a difference diagonal.
    # Similar calculation structure as for U_sum_val, using Inclusion-Exclusion for each relevant difference diagonal 'd'.
    U_diff_val = 0
    for d in D_set:
        # Determine the range of row indices 'i' for points (i, i-d) on the difference diagonal d within the NxN grid.
        # A point (i, j) is on diagonal d if i-j = d. It's within the grid if 1 <= i <= N and 1 <= j <= N.
        # Substituting j = i-d gives 1 <= i-d <= N.
        # Combining 1 <= i <= N and 1 <= i-d <= N yields the valid range for i:
        min_i = max(1, d + 1)
        max_i = min(N, N + d)
         
        # If min_i > max_i, the diagonal segment has no points within the grid. Skip this d.
        if min_i > max_i: 
            continue

        # Calculate the total number of integer points on this diagonal segment within the grid bounds.
        Pd_count = max_i - min_i + 1

        # Calculate adjustments using Inclusion-Exclusion.
        Cnt_A = 0 # Counts points (i, i-d) on the segment where the row index i is in A_set.
        for r_idx in A_set:
            if min_i <= r_idx <= max_i:
                 Cnt_A += 1
        
        Cnt_B = 0 # Counts points (i, i-d) on the segment where the column index j=i-d is in B_set.
        for c_idx in B_set:
             # Calculate the corresponding row index i for column index j=c_idx on diagonal d.
             i = c_idx + d 
             # Check if this row index i is within the valid range [min_i, max_i].
             if min_i <= i <= max_i:
                  Cnt_B += 1

        Cnt_AB = 0 # Counts points (i, i-d) where i is in A_set AND j=i-d is in B_set.
        for r_idx in A_set:
             # Check if row index i=r_idx is within the valid range...
             if min_i <= r_idx <= max_i:
                 # ...and if the corresponding column index j=r_idx-d is in B_set.
                 if (r_idx - d) in B_set:
                      Cnt_AB += 1

        # Apply Inclusion-Exclusion for diagonal d.
        Cd = Pd_count - Cnt_A - Cnt_B + Cnt_AB
        # Accumulate this count for diagonal d into the total U_diff_val.
        U_diff_val += Cd

    # Calculate |U_sum intersect U_diff|, the count of points (i, j) such that
    # i not in A_set, j not in B_set, i+j in S_set, and i-j in D_set.
    # These points lie at the intersection of an attacked sum diagonal and an attacked difference diagonal,
    # and are also safe from row/column captures.
    # Use a set to store unique intersection points found to avoid potential double counting.
    intersection_points = set() 
    
    # Iterate through all pairs of (s, d) from the sets S_set and D_set.
    for s in S_set:
        for d in D_set:
            # An intersection point (i, j) satisfies i+j=s and i-j=d.
            # For integer coordinates (i, j), s and d must have the same parity (both even or both odd).
            if (s + d) % 2 != 0: 
                continue # Skip if parity differs.

            # Calculate potential integer coordinates of the intersection point.
            i = (s + d) // 2
            j = (s - d) // 2

            # Check if the calculated point (i, j) lies within the NxN grid boundaries.
            if 1 <= i <= N and 1 <= j <= N:
                 # Check if the point is safe from row captures (i is not in A_set) 
                 # AND safe from column captures (j is not in B_set).
                 if i not in A_set and j not in B_set:
                    # If all conditions are met, this point (i, j) belongs to the intersection set U_sum intersect U_diff.
                    # Add the tuple (i, j) representing the point to the set. Sets automatically handle uniqueness.
                    intersection_points.add((i,j)) 

    # The size of the intersection set |U_sum intersect U_diff| is the number of unique points found.
    U_intersect_val = len(intersection_points) 

    # Final Answer Calculation:
    # The total number of safe squares is |Safe| = |G'| - |U_diag|,
    # where |G'| is the count of squares safe from row/column captures (TotalSafe_initial),
    # and |U_diag| is the count of squares in G' that are captured by diagonals.
    # By Principle of Inclusion-Exclusion: |U_diag| = |U_sum union U_diff| = |U_sum| + |U_diff| - |U_sum intersect U_diff|.
    # Substituting the computed values:
    # |Safe| = TotalSafe_initial - (U_sum_val + U_diff_val - U_intersect_val)
    FinalAnswer = TotalSafe_initial - (U_sum_val + U_diff_val - U_intersect_val)
    
    # Print the final calculated answer to standard output.
    print(FinalAnswer)

# Execute the solve function to run the program.
solve()