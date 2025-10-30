# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
adj_matrix = [list(map(int, data[i].split())) for i in range(1, N+1)]

for i in range(N):
    connected_vertices = [j+1 for j in range(N) if adj_matrix[i][j] == 1]
    print(" ".join(map(str, connected_vertices)))