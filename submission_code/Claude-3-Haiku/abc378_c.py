# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize the output list B
B = [-1] * N

# Iterate through the input list A
for i in range(N):
    # Find the most recent position before i where an element equal to A[i] appeared
    for j in range(i-1, -1, -1):
        if A[j] == A[i]:
            B[i] = j
            break

# Print the output list B
print(" ".join(map(str, B)))