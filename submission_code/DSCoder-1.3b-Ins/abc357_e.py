import sys

def solve():
    N = int(sys.stdin.readline().strip())
    a = list(map(int, sys.stdin.readline().strip().split()))

    graph = [[] for _ in range(N+1)]
    for i in range(N-1):
        u, v = a[i], a[i+1]
        graph[u].append(v)
        graph[v].append(u)

    count = 0
    for u in range(1, N+1):
        visited = [False] * (N+1)
        stack = [u]
        while stack:
            v = stack.pop()
            visited[v] = True
            if v == a[0]:
                count += 1
            for w in graph[v]:
                if not visited[w]:
                    stack.append(w)

    print(count)

solve()