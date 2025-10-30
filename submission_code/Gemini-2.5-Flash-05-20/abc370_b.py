# YOUR CODE HERE
import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())

    # Initialize a list of lists to store the combination matrix A.
    # A[row_0_based][col_0_based] corresponds to A_{row_0_based+1, col_0_based+1}
    # from the problem description (1-based indexing).
    # Since A_{i,j} is given only for i >= j, the matrix will be lower triangular.
    A = []
    for _ in range(N):
        # Each line provides elements for a specific row.
        # For a 0-indexed row 'r', there are 'r+1' elements.
        row_elements = list(map(int, sys.stdin.readline().split()))
        A.append(row_elements)

    # Initialize the current element to 1, as specified by the problem.
    current_element = 1

    # Iterate through elements 1, 2, ..., N in order to combine with current_element.
    for target_element in range(1, N + 1):
        # Let the two elements being combined be e1 and e2.
        e1 = current_element
        e2 = target_element

        # The problem states:
        # - If i >= j, they transform into A_{i, j}
        # - If i < j, they transform into A_{j, i}
        # This simplifies to always looking up A_{max(i, j), min(i, j)}.
        
        # Determine the 1-based indices for lookup in the conceptual A matrix.
        idx1_based_for_A = max(e1, e2)
        idx2_based_for_A = min(e1, e2)

        # Convert these 1-based indices to 0-based indices for accessing our Python list 'A'.
        # A_{x, y} (1-based) is stored at A[x-1][y-1] (0-based).
        row_0_based = idx1_based_for_A - 1
        col_0_based = idx2_based_for_A - 1

        # Get the result of the combination from the A matrix.
        result_of_combination = A[row_0_based][col_0_based]
        
        # Update current_element for the next iteration.
        current_element = result_of_combination

    # Print the final element obtained after all combinations.
    sys.stdout.write(str(current_element) + "
")

# Call the solve function to execute the program.
if __name__ == '__main__':
    solve()