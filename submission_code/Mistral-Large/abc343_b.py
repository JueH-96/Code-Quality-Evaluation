import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
adj_matrix = []
index = 1

for i in range(N):
    row = list(map(int, data[index:index + N]))
    adj_matrix.append(row)
    index += N

for i in range(N):
    connected_vertices = [j + 1 for j in range(N) if adj_matrix[i][j] == 1]
    if connected_vertices:
        print(" ".join(map(str, connected_vertices)))
    else:
        print()