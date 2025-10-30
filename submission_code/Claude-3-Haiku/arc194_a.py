import sys

# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize the sequence S
S = []
max_sum = 0

for a in A:
    if S:
        # If S is not empty, we can choose to delete the last element
        S.pop()
        max_sum = max(max_sum, sum(S) + a)
        S.append(a)
    else:
        # If S is empty, we must append the current element
        S.append(a)
        max_sum = max(max_sum, a)

# Print the answer
print(max_sum)