# YOUR CODE HERE
n, m = map(int, input().split())
friendships = []
for _ in range(m):
    a, b = map(int, input().split())
    friendships.append((a, b))

adj = [[] for _ in range(n + 1)]
for a, b in friendships:
    adj[a].append(b)
    adj[b].append(a)

count = 0
for i in range(1, n + 1):
    for j in range(len(adj[i])):
        for k in range(j + 1, len(adj[i])):
            friend1 = adj[i][j]
            friend2 = adj[i][k]
            if friend1 not in adj[friend2]:
                count += 1

print(count)