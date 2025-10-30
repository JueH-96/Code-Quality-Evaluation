# YOUR CODE HERE
n, m = map(int, input().split())
edges = []
for i in range(m):
    a, b, x, y = map(int, input().split())
    edges.append((a, b, x, y))

coords = [[0, 0] for _ in range(n + 1)]
graph = [[] for _ in range(n + 1)]
for a, b, x, y in edges:
    graph[a].append((b, x, y))

visited = [False] * (n + 1)
stack = [1]
visited[1] = True

while stack:
    curr = stack.pop()
    for neighbor, x, y in graph[curr]:
        if not visited[neighbor]:
            coords[neighbor][0] = coords[curr][0] + x
            coords[neighbor][1] = coords[curr][1] + y
            visited[neighbor] = True
            stack.append(neighbor)

undecidable = [False] * (n + 1)
for i in range(1, n + 1):
    if not visited[i]:
        undecidable[i] = True

for i in range(1, n + 1):
    if undecidable[i]:
        print('undecidable')
    else:
        print(coords[i][0], coords[i][1])