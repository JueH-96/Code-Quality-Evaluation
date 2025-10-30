# YOUR CODE HERE
n = int(input())
A = list(map(int, input().split()))

while True:
    # Check if all adjacent pairs have absolute difference of 1
    all_diff_one = True
    first_bad_index = -1
    
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            all_diff_one = False
            first_bad_index = i
            break
    
    if all_diff_one:
        break
    
    # Found a pair with difference != 1
    i = first_bad_index
    A_i = A[i]
    A_i_plus_1 = A[i+1]
    
    # Create the list of numbers to insert
    to_insert = []
    if A_i < A_i_plus_1:
        # Insert A_i+1, A_i+2, ..., A_{i+1}-1
        for num in range(A_i + 1, A_i_plus_1):
            to_insert.append(num)
    else:  # A_i > A_i_plus_1
        # Insert A_i-1, A_i-2, ..., A_{i+1}+1
        for num in range(A_i - 1, A_i_plus_1, -1):
            to_insert.append(num)
    
    # Insert the numbers
    A = A[:i+1] + to_insert + A[i+1:]

# Print the result
print(' '.join(map(str, A)))