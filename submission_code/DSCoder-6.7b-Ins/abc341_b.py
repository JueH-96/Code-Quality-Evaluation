import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))
ST = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N-1)]

# Sort the list of operations in descending order of S_i
ST.sort(key=lambda x: -x[0])

# Initialize the maximum number of units of the currency of country N
max_units = A[-1]

# Perform the operations
for i in range(N-1):
    S, T = ST[i]
    # Calculate the number of operations that can be performed
    operations = A[i] // S
    # Update the number of units of the currency of country (i+1)
    A[i+1] += min(operations * T, A[i+1])
    # Update the maximum number of units of the currency of country N
    max_units = max(max_units, A[i+1])

print(max_units)