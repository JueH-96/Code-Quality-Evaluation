# YOUR CODE HERE
n = int(input())
adj_matrix = []
for _ in range(n):
    adj_matrix.append(list(map(int, input().split())))

for i in range(n):
    connected = []
    for j in range(n):
        if adj_matrix[i][j] == 1 and i != j:
            connected.append(str(j+1))
    print(' '.join(connected))