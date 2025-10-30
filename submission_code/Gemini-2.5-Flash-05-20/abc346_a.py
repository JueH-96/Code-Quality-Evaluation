# Read the integer N
N = int(input())

# Read the N integers A as a list
A = list(map(int, input().split()))

# Initialize an empty list to store the B values
B_values = []

# Calculate B_i = A_i * A_{i+1} for 1 <= i <= N-1
# In 0-indexed Python lists, this means A[j] * A[j+1] for j from 0 to N-2
for i in range(N - 1):
    product = A[i] * A[i+1]
    B_values.append(product)

# Print the B values separated by spaces
# map(str, B_values) converts each integer in B_values to a string
# ' '.join(...) joins these strings with a space in between
print(' '.join(map(str, B_values)))