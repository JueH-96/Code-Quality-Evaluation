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
        v, k = query[1] - 1, query[2]
        connected = sorted(graph[v] + [v], reverse=True)
        if k > len(connected):
            print(-1)
        else:
            print(connected[k - 1] + 1)