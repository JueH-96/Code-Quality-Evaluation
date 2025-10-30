# YOUR CODE HERE
import sys
from collections import deque

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = data[N+2:]

# Initialize a dictionary to keep track of the index of each element in A
index_map = {A[i]: i for i in range(len(A))}

# Process each query
for i in range(0, 2*Q, 2):
    query_type = int(queries[i])
    x = int(queries[i+1])
    
    if query_type == 1:
        y = int(queries[i+2])
        idx = index_map[x]
        A.insert(idx + 1, y)
        # Update the index map
        for j in range(idx + 1, len(A)):
            index_map[A[j]] = j
    elif query_type == 2:
        idx = index_map[x]
        A.pop(idx)
        # Update the index map
        for j in range(idx, len(A)):
            index_map[A[j]] = j
        del index_map[x]

# Print the final sequence
print(" ".join(map(str, A)))