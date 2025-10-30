import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    edges = []
    for _ in range(M):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        edges.append((u, v))

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def bfs(start):
        queue = deque([start])
        visited = set([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    components = []
    visited = set()
    for i in range(1, N + 1):
        if i not in visited:
            component = bfs(i)
            components.append(component)
            visited.update(component)

    def is_valid_component(component):
        degree = {v: 0 for v in component}
        for v in component:
            for neighbor in graph[v]:
                if neighbor in component:
                    degree[v] += 1

        odd_degree_count = sum(1 for count in degree.values() if count % 2 == 1)
        return odd_degree_count == 0

    for component in components:
        if not is_valid_component(component):
            print("No")
            return

    print("Yes")
    result = [1] * N
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    solve()