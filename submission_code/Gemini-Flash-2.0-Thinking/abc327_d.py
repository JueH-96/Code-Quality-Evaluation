from collections import deque

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    adj = [[] for _ in range(n)]
    for i in range(m):
        u, v = a[i] - 1, b[i] - 1
        if u == v:
            print("No")
            return
        adj[u].append(v)
        adj[v].append(u)

    color = [-1] * n

    for start_node in range(n):
        if color[start_node] == -1:
            queue = deque([(start_node, 0)])
            color[start_node] = 0

            while queue:
                u, c = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - c
                        queue.append((v, 1 - c))
                    elif color[v] == c:
                        print("No")
                        return

    print("Yes")

solve()