# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Number of vertices
N = int(data[0])

# Adjacency matrix
A = []
index = 1
for i in range(N):
    row = list(map(int, data[index:index + N]))
    A.append(row)
    index += N

# Print the directly connected vertices for each vertex
for i in range(N):
    connected = [j + 1 for j in range(N) if A[i][j] == 1]
    print(" ".join(map(str, connected)))