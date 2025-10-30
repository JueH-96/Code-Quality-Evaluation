# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
adj_matrix = [list(map(int, input[i*N+1:(i+1)*N+1])) for i in range(N)]

for i in range(N):
    connected_vertices = [j+1 for j in range(N) if adj_matrix[i][j] == 1]
    print(" ".join(map(str, connected_vertices)))