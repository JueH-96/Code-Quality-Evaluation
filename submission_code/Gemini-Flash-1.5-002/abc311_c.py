# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

visited = [False] * (n + 1)
path = []
cycle = []

def dfs(u):
    visited[u] = True
    path.append(u)
    v = a[u - 1]
    if visited[v]:
        if v in path:
            index = path.index(v)
            cycle.extend(path[index:])
            return True
    else:
        if dfs(v):
            return True
    path.pop()
    return False

for i in range(1, n + 1):
    if not visited[i]:
        if dfs(i):
            break

print(len(cycle))
print(*cycle)