# YOUR CODE HERE

N = int(input())
adj_matrix = []
for _ in range(N):
    row = list(map(int, input().split()))
    adj_matrix.append(row)

for i in range(N):
    connected_vertices = [j+1 for j in range(N) if adj_matrix[i][j] == 1]
    print(*connected_vertices)