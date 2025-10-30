# YOUR CODE HERE
def is_good_pair(N, M, A, B):
    graph = [[] for _ in range(N + 1)]
    for a, b in zip(A, B):
        graph[a].append(b)
        graph[b].append(a)

    color = [0] * (N + 1)

    def dfs(node, c):
        color[node] = c
        for neighbor in graph[node]:
            if color[neighbor] == c:
                return False
            if color[neighbor] == 0 and not dfs(neighbor, 3 - c):
                return False
        return True

    for i in range(1, N + 1):
        if color[i] == 0:
            if not dfs(i, 1):
                return False
    return True

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print("Yes" if is_good_pair(N, M, A, B) else "No")