import sys

def solve():
    N, M = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]
    if M == 0:
        print("Yes")
        print(1)
        return

    # Create adjacency list
    adj_list = [[] for _ in range(N)]
    for u, v in edges:
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)

    # Check if the graph is bipartite
    color = [-1] * N
    for i in range(N):
        if color[i] == -1:
            color[i] = 0
            stack = [i]
            while stack:
                v = stack.pop()
                for u in adj_list[v]:
                    if color[u] == -1:
                        color[u] = 1 - color[v]
                        stack.append(u)
                    elif color[u] == color[v]:
                        print("No")
                        return

    # Assign values to vertices
    values = [1 << i for i in range(N)]
    print("Yes")
    print(*values)

solve()