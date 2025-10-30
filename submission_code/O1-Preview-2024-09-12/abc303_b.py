# YOUR CODE HERE
N, M = map(int, input().split())
N = int(N)
M = int(M)
adj = [[False]*N for _ in range(N)]
for _ in range(M):
    a = [int(x)-1 for x in input().split()]
    for j in range(N-1):
        x = a[j]
        y = a[j+1]
        adj[x][y] = True
        adj[y][x] = True
count = 0
for x in range(N):
    for y in range(x+1, N):
        if not adj[x][y]:
            count +=1
print(count)