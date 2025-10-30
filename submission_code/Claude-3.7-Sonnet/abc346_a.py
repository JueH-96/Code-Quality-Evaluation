# Read the integer N
N = int(input())

# Read the N integers
A = list(map(int, input().split()))

# Calculate B_i = A_i * A_{i+1} for each i from 1 to N-1
B = [A[i] * A[i+1] for i in range(N-1)]

# Print B_1, B_2, ..., B_{N-1} separated by spaces
print(" ".join(map(str, B)))