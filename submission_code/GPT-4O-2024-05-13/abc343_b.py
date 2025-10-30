# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().strip().split()
N = int(data[0])
matrix = []
index = 1

for i in range(N):
    row = []
    for j in range(N):
        row.append(int(data[index]))
        index += 1
    matrix.append(row)

for i in range(N):
    connected_vertices = []
    for j in range(N):
        if matrix[i][j] == 1:
            connected_vertices.append(j + 1)
    print(" ".join(map(str, connected_vertices)))