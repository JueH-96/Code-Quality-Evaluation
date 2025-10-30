n, k = map(int, input().split())
p = list(map(int, input().split()))
visited = [False] * (n + 1)
ans = [0] * (n + 1)

for x in range(1, n + 1):
    if not visited[x]:
        cycle = []
        current = x
        while True:
            if visited[current]:
                break
            visited[current] = True
            cycle.append(current)
            current = p[current - 1]  # Convert to 0-based index for p
        L = len(cycle)
        for i in range(L):
            original = cycle[i]
            new_i = (i + k) % L
            ans[original] = cycle[new_i]

print(' '.join(map(str, ans[1:n+1])))