# YOUR CODE HERE
n, m = map(int, input().split())
s = [input() for _ in range(n)]
visited = [False] * n
flavors = [False] * m

for i in range(n):
    for j in range(m):
        if s[i][j] == 'o':
            flavors[j] = True

for i in range(n):
    if all(s[i][j] == 'o' or flavors[j] for j in range(m)):
        visited[i] = True

print(sum(visited))