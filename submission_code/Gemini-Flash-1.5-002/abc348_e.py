import collections

def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u - 1, v - 1))
    c = list(map(int, input().split()))

    graph = collections.defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def bfs(start_node):
        distances = [-1] * n
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            curr = queue.popleft()
            for neighbor in graph[curr]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances

    min_f = float('inf')
    for i in range(n):
        distances = bfs(i)
        f_i = sum(c[j] * distances[j] for j in range(n))
        min_f = min(min_f, f_i)

    print(min_f)

solve()