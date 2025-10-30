# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

edges = []
for i in range(m):
    edges.append((a[i], b[i]))

graph = {}
for u, v in edges:
    if u not in graph:
        graph[u] = []
    if v not in graph:
        graph[v] = []
    graph[u].append(v)
    graph[v].append(u)

color = {}
possible = True
for i in range(1, n + 1):
    if i not in graph:
        continue
    if i not in color:
        stack = [i]
        color[i] = 0
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    stack.append(v)
                elif color[v] == color[u]:
                    possible = False
                    break
            if not possible:
                break
    if not possible:
        break

if possible:
    print("Yes")
else:
    print("No")