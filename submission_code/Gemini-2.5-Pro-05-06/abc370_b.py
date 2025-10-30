# Read N, the number of types of elements.
N = int(input())

# Read the transformation matrix A.
# A_storage[i] will store the (i+1)-th row of input values,
# which correspond to A_{i+1,1}, A_{i+1,2}, ..., A_{i+1,i+1}.
# These are 1-indexed element numbers.
# In A_storage, A_storage[i][j] will store A_{i+1, j+1}.
A_storage = []
for _ in range(N):
    row_values = list(map(int, input().split()))
    A_storage.append(row_values)

# Initialize the current element. We start with element 1.
# Element numbers are 1-indexed as per the problem statement.
current_element = 1

# Sequentially combine the current_element with elements 1, 2, ..., N.
# The loop variable k_to_combine will take values 1, 2, ..., N.
for k_to_combine in range(1, N + 1):
    # The two elements being combined are `current_element` and `k_to_combine`.
    # Both are 1-indexed element numbers.
    
    # Determine the 1-based row and column indices for looking up in matrix A.
    # The rule is: result is A_{i,j} if i >= j, and A_{j,i} if i < j.
    # This is equivalent to A_{max(element1, element2), min(element1, element2)}.
    row_1_based = max(current_element, k_to_combine)
    col_1_based = min(current_element, k_to_combine)
    
    # Retrieve the new element's value from A_storage.
    # A_storage is 0-indexed. A_storage[x][y] stores A_{x+1, y+1}.
    # So, to get A_{row_1_based, col_1_based}, we use indices (row_1_based - 1) 
    # for the outer list and (col_1_based - 1) for the inner list.
    new_element_value = A_storage[row_1_based - 1][col_1_based - 1]
    
    # Update current_element to the new value.
    current_element = new_element_value
    
# Print the final element obtained after all combinations.
print(current_element)