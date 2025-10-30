n = int(input())
adj_matrix = []
for _ in range(n):
    adj_matrix.append(list(map(int, input().split())))

for i in range(n):
    connected_vertices = []
    for j in range(n):
        if adj_matrix[i][j] == 1:
            connected_vertices.append(j + 1)
    print(*connected_vertices)