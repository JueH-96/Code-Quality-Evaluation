n, k = map(int, input().split())
p = [0] + list(map(int, input().split()))
visited = [False] * (n + 1)
result = [0] * (n + 1)

for i in range(1, n + 1):
    if not visited[i]:
        cycle = []
        current = i
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = p[current]
        l = len(cycle)
        m = pow(2, k, l)
        for idx in range(l):
            elem = cycle[idx]
            new_idx = (idx + m) % l
            result[elem] = cycle[new_idx]

print(' '.join(map(str, result[1:])))