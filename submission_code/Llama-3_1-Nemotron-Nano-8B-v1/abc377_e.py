n, k = map(int, input().split())
p = list(map(int, input().split()))
visited = [False] * (n + 1)
mapping = {}

for i in range(1, n + 1):
    if not visited[i]:
        cycle = []
        current = i
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = p[current - 1]
        l = len(cycle)
        effective = k % l
        for idx, x in enumerate(cycle):
            new_idx = (idx + effective) % l
            mapping[x] = cycle[new_idx]

print(' '.join(str(mapping[i]) for i in range(1, n + 1)))