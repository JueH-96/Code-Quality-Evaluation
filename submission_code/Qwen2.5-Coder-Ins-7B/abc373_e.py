# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

# Calculate the total number of votes received so far
total_votes = sum(A)

# Calculate the number of votes left to be counted
remaining_votes = K - total_votes

# Initialize the result list
C = [-1] * N

# Iterate over each candidate
for i in range(N):
    # If the candidate has already secured their victory
    if A[i] >= remaining_votes + M - 1:
        C[i] = 0
    else:
        # Calculate the minimum number of additional votes needed
        C[i] = remaining_votes + M - 1 - A[i]

# Print the result
print(" ".join(map(str, C)))