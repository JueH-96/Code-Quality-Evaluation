# YOUR CODE HERE
n, q = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1] - 1, query[2] - 1
        graph[u].append(v)
        graph[v].append(u)
    else:
        v = query[1] - 1
        for u in graph[v]:
            graph[u].remove(v)
        graph[v] = []
    count = 0
    for i in range(n):
        if len(graph[i]) == 0:
            count += 1
    print(count)