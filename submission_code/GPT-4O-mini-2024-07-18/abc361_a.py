# YOUR CODE HERE
def insert_after_kth_element(N, K, X, A):
    # Create a new list B by inserting X after the K-th element of A
    B = A[:K] + [X] + A[K:]
    return B

import sys

# Read input
input_data = sys.stdin.read().strip().splitlines()
first_line = input_data[0].split()
N = int(first_line[0])
K = int(first_line[1])
X = int(first_line[2])
A = list(map(int, input_data[1].split()))

# Get the result
result = insert_after_kth_element(N, K, X, A)

# Print the result
print(' '.join(map(str, result)))