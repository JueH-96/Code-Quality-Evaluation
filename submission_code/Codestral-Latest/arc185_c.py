# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
A = list(map(int, data[2:]))

# Create a dictionary to store the indices of each element
index_dict = {}
for i in range(N):
    index_dict[A[i]] = i

# Iterate through all pairs of elements
for i in range(N):
    for j in range(i + 1, N):
        # Calculate the complement needed to reach X
        complement = X - A[i] - A[j]
        # Check if the complement exists in the dictionary and is not one of the current indices
        if complement in index_dict and index_dict[complement] > j:
            print(i + 1, j + 1, index_dict[complement] + 1)
            sys.exit()

# If no such triple is found, print -1
print(-1)