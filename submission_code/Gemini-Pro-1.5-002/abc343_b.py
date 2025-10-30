# YOUR CODE HERE
n = int(input())
adj = []
for _ in range(n):
    adj.append(list(map(int, input().split())))

for i in range(n):
    connected = []
    for j in range(n):
        if adj[i][j] == 1:
            connected.append(j + 1)
    if connected:
        print(*connected)