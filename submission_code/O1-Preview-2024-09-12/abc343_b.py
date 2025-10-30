# YOUR CODE HERE
N = int(input())
adj_matrix = []
for _ in range(N):
    adj_matrix.append(list(map(int, input().split())))
for i in range(N):
    connected = []
    for j in range(N):
        if adj_matrix[i][j] == 1:
            connected.append(j+1)  # vertices are labeled from 1 to N
    print(' '.join(map(str, connected)))