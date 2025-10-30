# YOUR CODE HERE
n, m, x1 = map(int, input().split())
trains = []
for i in range(m):
    a, b, s, t = map(int, input().split())
    trains.append((a, b, s, t))

x = [0] * m
x[0] = x1

graph = {}
for i in range(m):
    if trains[i][1] not in graph:
        graph[trains[i][1]] = []
    graph[trains[i][1]].append(i)

for i in range(m):
    for j in range(m):
        if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
            x[j] = max(x[j], x[i] + trains[j][2] - trains[i][3])

print(*x[1:])