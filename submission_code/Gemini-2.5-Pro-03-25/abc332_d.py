# YOUR CODE HERE
import sys
import itertools

# Function to calculate inversions in a permutation
def count_inversions(p):
    """
    Calculates the number of inversions in a permutation p.
    An inversion is a pair of indices (i, j) such that i < j and p[i] > p[j].
    The number of inversions is equal to the minimum number of adjacent swaps 
    required to sort the permutation into identity order, or equivalently, 
    to obtain the permutation p from the identity permutation.
    
    Args:
        p: A tuple or list representing the permutation. 
           Example: (0, 2, 1) is a permutation of (0, 1, 2).

    Returns:
        The number of inversions in p.
    """
    count = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if the pair (i, j) constitutes an inversion
            if p[i] > p[j]:
                count += 1
    return count

def solve():
    """
    Reads the input grids A and B, finds the minimum number of adjacent row/column swaps 
    to transform A into B, and prints the result. If transformation is impossible, prints -1.
    """
    # Read dimensions H (height/rows) and W (width/columns) from standard input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read grid A from standard input
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))

    # Read grid B from standard input
    B = []
    for _ in range(H):
        B.append(list(map(int, sys.stdin.readline().split())))

    # Initialize minimum total swaps required to positive infinity
    min_total_swaps = float('inf')
    
    # Generate lists of row and column indices (0-based)
    row_indices = list(range(H))
    col_indices = list(range(W))

    # Generate an iterator for all possible permutations of row indices
    # Each permutation `pi` represents a potential final ordering of rows from grid A.
    # Specifically, `pi[r]` gives the index of the row from the original grid A 
    # that ends up at row `r` in the transformed grid.
    row_perms_iter = itertools.permutations(row_indices)
    
    # Generate all possible permutations of column indices and store them in a list.
    # This is done because we need to iterate through all column permutations for each row permutation.
    # Since W <= 5, W! <= 120, storing them in a list is feasible and efficient.
    # Each permutation `sigma` represents a potential final ordering of columns from grid A.
    # `sigma[c]` gives the index of the column from the original grid A 
    # that ends up at column `c` in the transformed grid.
    col_perms_list = list(itertools.permutations(col_indices)) 

    # Flag to track if at least one valid transformation is found
    found_match = False

    # Iterate through all possible row permutations pi
    for pi in row_perms_iter: 
        
        # Calculate the minimum number of adjacent swaps (inversions) required for the row permutation pi
        inv_pi = count_inversions(pi)

        # Optimization: If the number of row swaps alone is already greater than or equal to 
        # the minimum total swaps found so far, then this path cannot lead to a better solution.
        # Skip checking column permutations for this `pi`.
        if inv_pi >= min_total_swaps:
            continue

        # Iterate through all possible column permutations sigma
        for sigma in col_perms_list: 
            
            # Calculate the minimum number of adjacent swaps (inversions) required for the column permutation sigma
            inv_sigma = count_inversions(sigma)
            
            # Optimization: Check if the current combination of row and column swaps 
            # already requires more swaps than the best solution found so far.
            if inv_pi + inv_sigma >= min_total_swaps:
                 continue

            # Check if applying the current row permutation `pi` and column permutation `sigma`
            # to grid A results in a grid identical to grid B.
            match = True
            # Iterate through each cell (r, c) of the target grid B
            for r in range(H): 
                for c in range(W): 
                    # According to the permutations, the element at cell (r, c) in the transformed grid A 
                    # should come from the original grid A at cell (pi[r], sigma[c]).
                    # Check if this element matches the element at cell (r, c) in grid B.
                    if A[pi[r]][sigma[c]] != B[r][c]:
                        match = False
                        # If a mismatch is found, this pair of permutations (pi, sigma) is invalid.
                        # Break the inner loop (over columns c).
                        break 
                if not match:
                    # If a mismatch was found in the inner loop, break the outer loop (over rows r) as well.
                    break 
            
            # If the loop completed without finding any mismatch, the grids match.
            if match:
                found_match = True # Mark that at least one valid transformation is found.
                # Calculate the total number of swaps for this valid transformation.
                current_total_swaps = inv_pi + inv_sigma
                # Update the overall minimum total swaps required.
                min_total_swaps = min(min_total_swaps, current_total_swaps)

    # After checking all possible pairs of row and column permutations:
    if not found_match:
        # If no combination of permutations could transform A into B, it's impossible. Print -1.
        print("-1")
    else:
        # Otherwise, print the minimum total number of swaps found.
        print(min_total_swaps)

# Execute the main solve function when the script is run
if __name__ == '__main__':
    solve()