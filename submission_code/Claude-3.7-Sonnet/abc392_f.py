# Read the number of operations
N = int(input())
# Read the positions for each insertion
P = list(map(int, input().split()))

# Start with an empty array
A = []

# Perform the operations
for i in range(1, N+1):
    # Insert i at the position P_i-1 (0-indexed for Python)
    A.insert(P[i-1]-1, i)

# Print the final array with space separation
print(" ".join(map(str, A)))