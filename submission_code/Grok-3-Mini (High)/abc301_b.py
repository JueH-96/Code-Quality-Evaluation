import sys

# Read input
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:1 + N]))

# Apply the procedure until all adjacent differences are 1
while True:
    # Find the first index where absolute difference is not 1
    idx = -1
    for i in range(len(A) - 1):
        if abs(A[i] - A[i + 1]) != 1:
            idx = i
            break
    
    # If no such index, all differences are 1, break
    if idx == -1:
        break
    
    # Insert elements based on comparison
    if A[idx] < A[idx + 1]:
        new_elements = list(range(A[idx] + 1, A[idx + 1]))
    else:  # A[idx] > A[idx + 1]
        new_elements = list(range(A[idx] - 1, A[idx + 1], -1))
    
    # Insert the new elements into the list
    A = A[:idx + 1] + new_elements + A[idx + 1:]

# Print the final sequence separated by spaces
print(' '.join(map(str, A)))