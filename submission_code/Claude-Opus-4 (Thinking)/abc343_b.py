# YOUR CODE HERE
n = int(input())
adj_matrix = []
for i in range(n):
    row = list(map(int, input().split()))
    adj_matrix.append(row)

for i in range(n):
    connected = []
    for j in range(n):
        if adj_matrix[i][j] == 1:
            connected.append(j + 1)  # Convert to 1-based vertex number
    if connected:
        print(' '.join(map(str, connected)))
    else:
        print()  # Empty line if no connections