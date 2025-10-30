# YOUR CODE HERE
n, m = map(int, input().split())
photos = []
for i in range(m):
    photos.append(list(map(int, input().split())))

adj = [[False] * (n + 1) for _ in range(n + 1)]
for photo in photos:
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == 1:
                adj[photo[i]][photo[j]] = True
                adj[photo[j]][photo[i]] = True

count = 0
for i in range(1, n + 1):
    for j in range(i + 1, n + 1):
        if not adj[i][j]:
            count += 1

print(count)