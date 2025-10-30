import sys
from bisect import bisect_left

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

# Sort the votes in descending order
A.sort(reverse=True)

# Calculate the remaining votes
remaining_votes = K - sum(A)

# Initialize the result list
result = []

for i in range(N):
    # If the candidate has already secured their victory
    if i < M and A[i] > A[M]:
        result.append(0)
    else:
        # Calculate the minimum additional votes needed
        needed_votes = A[M] - A[i] + 1
        if needed_votes <= remaining_votes:
            result.append(needed_votes)
        else:
            result.append(-1)

# Print the result
print(" ".join(map(str, result)))