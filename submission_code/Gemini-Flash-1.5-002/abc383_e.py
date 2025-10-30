import itertools

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    def find_min_path_weight(start, end):
        q = [(start, [start], 0)]  # (node, path, max_weight)
        visited = set()
        min_weight = float('inf')

        while q:
            curr, path, max_weight = q.pop(0)
            if curr == end:
                min_weight = min(min_weight, max_weight)
                continue
            visited.add(curr)

            for neighbor, weight in graph[curr]:
                if neighbor not in visited:
                    new_max_weight = max(max_weight, weight)
                    q.append((neighbor, path + [neighbor], new_max_weight))
        return min_weight

    min_sum = float('inf')
    for perm in itertools.permutations(b):
        curr_sum = 0
        for i in range(k):
            curr_sum += find_min_path_weight(a[i], perm[i])
        min_sum = min(min_sum, curr_sum)

    print(min_sum)

solve()