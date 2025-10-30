# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import defaultdict

data = input().split()
N = int(data[0])
Q = int(data[1])
A = list(map(int, data[2:N+2]))
queries = data[N+2:]

# Function to find the mex of a list
def find_mex(A):
    seen = set(A)
    mex = 0
    while mex in seen:
        mex += 1
    return mex

# Process each query
index = 0
for _ in range(Q):
    i_k = int(queries[index]) - 1
    x_k = int(queries[index + 1])
    A[i_k] = x_k
    print(find_mex(A))
    index += 2