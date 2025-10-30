# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input, performs the sequence of element combinations,
    and prints the final element.
    """
    # Read the number of element types, N
    N = int(sys.stdin.readline())

    # Read the combination matrix A.
    # The input provides A_ij for i >= j (1-indexed).
    # We store it in a list of lists where A_matrix[i-1][j-1] stores A_ij for i >= j (0-indexed).
    # Example:
    # A_matrix[0][0] stores A_1,1
    # A_matrix[1][0] stores A_2,1
    # A_matrix[1][1] stores A_2,2
    # A_matrix[i-1][j-1] stores A_i,j for i >= j
    A_matrix = []
    for i in range(1, N + 1):
        # Read the i-th row of the lower triangular matrix
        # This row contains A_i,1, A_i,2, ..., A_i,i
        row = list(map(int, sys.stdin.readline().split()))
        A_matrix.append(row)

    # The process starts with element 1. (1-indexed)
    current_element = 1

    # Combine the current element sequentially with elements 1, 2, ..., N. (1-indexed)
    for k in range(1, N + 1):
        # Combine current_element (curr) with element k.
        # The combination rule is A_i,j if i >= j, and A_j,i if i < j.
        # This is equivalent to looking up the value A_max(i,j), min(i,j).
        # Let i = current_element and j = k (both are 1-indexed element numbers).
        u = max(current_element, k) # 1-indexed row for A lookup
        v = min(current_element, k) # 1-indexed column for A lookup

        # We need the value of A_u,v where u >= v.
        # In our 0-indexed A_matrix, A_u,v is stored at index [u-1][v-1].
        # Row u (1-indexed) corresponds to A_matrix[u-1] (0-indexed).
        # Column v (1-indexed) within row u corresponds to index v-1 (0-indexed) in A_matrix[u-1].
        # A_matrix[u-1] contains values A_u,1, A_u,2, ..., A_u,u.
        # A_matrix[u-1][v-1] correctly accesses A_u,v since v <= u.
        current_element = A_matrix[u-1][v-1]

    # Print the final element obtained.
    print(current_element)

# Execute the solve function
solve()