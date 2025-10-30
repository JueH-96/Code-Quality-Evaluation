import sys

# Read the input
N = int(sys.stdin.readline().strip())
A = [int(x) for x in sys.stdin.readline().strip().split()]

# Solve the problem
B = []
for i in range(N):
    B_i = sum(x for x in A if x > A[i])
    B.append(B_i)

# Print the output
print(" ".join(str(b) for b in B))