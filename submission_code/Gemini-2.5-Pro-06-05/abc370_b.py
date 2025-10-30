import sys

# YOUR CODE HERE
def solve():
    """
    This function solves the problem by reading input, simulating the
    combination process, and printing the final result.
    """
    
    # Read N, the number of element types, from standard input.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This handles cases where input is empty or malformed.
        return

    # Read the lower triangular matrix A.
    # The problem uses 1-based indexing (elements 1 to N), while Python lists are 0-indexed.
    # We will store the matrix in a 0-indexed list of lists (A_matrix).
    # A_{r, c} from the problem (with r >= c) will be at A_matrix[r-1][c-1].
    A_matrix = []
    for _ in range(N):
        # Read each row, split by spaces, and convert to a list of integers.
        row = list(map(int, sys.stdin.readline().split()))
        A_matrix.append(row)

    # The process starts with element 1.
    current_element = 1

    # We combine the current element with elements j = 1, 2, ..., N in this order.
    for j in range(1, N + 1):
        # 'i' represents the element we have before this step's combination.
        i = current_element
        
        # The combination rule is defined as:
        # - If i >= j, the result is A_{i, j}.
        # - If i < j, the result is A_{j, i}.
        # This ensures we always access an element from the given lower triangular part.
        
        # Determine the new element based on the rule, adjusting for 0-based indexing.
        if i >= j:
            # Result is A_{i, j}, which corresponds to A_matrix[i-1][j-1].
            new_element = A_matrix[i - 1][j - 1]
        else:  # i < j
            # Result is A_{j, i}, which corresponds to A_matrix[j-1][i-1].
            new_element = A_matrix[j - 1][i - 1]
        
        # Update the current element for the next iteration.
        current_element = new_element

    # After the loop completes, current_element holds the final result.
    # Print the final element number to standard output.
    print(current_element)

solve()