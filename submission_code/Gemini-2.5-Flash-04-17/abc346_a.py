# Read N from stdin
N = int(input())

# Read the N integers into a list A from stdin
A = list(map(int, input().split()))

# Create a list to store the products B
B = []

# Calculate B_i = A_i * A_{i+1} for i from 1 to N-1
# Using 0-based indexing for list A, this corresponds to A[i] * A[i+1] for i from 0 to N-2
for i in range(N - 1):
    product = A[i] * A[i + 1]
    B.append(product)

# Print the elements of B separated by spaces to stdout
# The * operator unpacks the list B into arguments for print
print(*B)