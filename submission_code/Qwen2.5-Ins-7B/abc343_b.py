# YOUR CODE HERE
n = int(input())
adj_matrix = [list(map(int, input().split())) for _ in range(n)]
for i in range(n):
    connections = [j + 1 for j in range(n) if adj_matrix[i][j] == 1]
    print(*connections)