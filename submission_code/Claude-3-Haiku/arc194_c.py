# Read the input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Initialize the minimum cost
min_cost = 0

# Iterate through the sequences
for i in range(N):
    # If the values are different, flip the value in A and add the cost
    if A[i] != B[i]:
        min_cost += C[i] * sum(A)
        A[i] = 1 - A[i]

# Print the minimum cost
print(min_cost)