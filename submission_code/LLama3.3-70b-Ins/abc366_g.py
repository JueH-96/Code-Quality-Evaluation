import sys

def read_ints():
    return map(int, sys.stdin.readline().split())

def read_int():
    return int(sys.stdin.readline())

def solve():
    N, M = read_ints()
    graph = [[] for _ in range(N)]
    for _ in range(M):
        u, v = read_ints()
        graph[u-1].append(v-1)
        graph[v-1].append(u-1)

    # Check if the graph is bipartite
    colors = [-1] * N
    for i in range(N):
        if colors[i] == -1:
            colors[i] = 0
            stack = [i]
            while stack:
                node = stack.pop()
                for neighbor in graph[node]:
                    if colors[neighbor] == -1:
                        colors[neighbor] = 1 - colors[node]
                        stack.append(neighbor)
                    elif colors[neighbor] == colors[node]:
                        print("No")
                        return

    # Assign values to nodes
    values = [1] * N
    for i in range(N):
        if colors[i] == 1:
            values[i] *= 2

    print("Yes")
    print(*values)

if __name__ == "__main__":
    solve()