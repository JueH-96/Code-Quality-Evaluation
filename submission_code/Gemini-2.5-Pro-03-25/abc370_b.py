# YOUR CODE HERE
import sys

if __name__ == '__main__':
    # Read N: the number of element types
    # N is an integer between 1 and 100.
    n = int(sys.stdin.readline())
    
    # Read the lower triangular matrix A representing combination results.
    # The input provides A_{i, j} for i >= j.
    # We store this data in a list of lists called `a_data`.
    # `a_data[i]` will store the input row corresponding to combinations involving element i+1.
    # Specifically, `a_data[i]` = [A_{i+1, 1}, A_{i+1, 2}, ..., A_{i+1, i+1}].
    # We use 0-based indexing for lists in Python. Therefore, the value A_{i+1, j+1} (where i >= j)
    # is stored at `a_data[i][j]`.
    a_data = []
    for i in range(n):
        # Read the (i+1)-th line of the matrix input, which contains i+1 integers.
        row = list(map(int, sys.stdin.readline().split()))
        a_data.append(row)

    # Initialize the starting element. The process begins with element 1.
    # We use 1-based numbering for elements as specified in the problem.
    current_element = 1

    # Simulate the sequence of N combinations as described in the problem.
    # We start with element 1, and combine it sequentially with elements 1, 2, ..., N.
    # The result of each combination becomes the element used in the next combination.
    for j in range(1, n + 1): # Iterate through the elements to combine with: j = 1, 2, ..., N
        # 'c' holds the element number we currently have before this combination step.
        c = current_element 
        
        # Determine the result of combining element 'c' and element 'j'
        # according to the rule:
        # If c >= j, the result is A_{c, j}.
        # If c < j, the result is A_{j, c}.
        
        if c >= j:
            # Access A_{c, j} from our stored data `a_data`.
            # Since element numbers 'c' and 'j' are 1-based, and list indices are 0-based,
            # A_{c, j} corresponds to the element at index `j-1` in the row `c-1`.
            # So, we access `a_data[c-1][j-1]`.
            result = a_data[c-1][j-1]
        else: # c < j
            # Access A_{j, c} from our stored data `a_data`.
            # Similarly, A_{j, c} corresponds to the element at index `c-1` in the row `j-1`.
            # So, we access `a_data[j-1][c-1]`.
            result = a_data[j-1][c-1]
            
        # Update the current element to the result of the combination.
        # This new element will be used in the next iteration of the loop.
        current_element = result

    # After completing all N combinations, `current_element` holds the final element number.
    # Print the final result to standard output.
    print(current_element)