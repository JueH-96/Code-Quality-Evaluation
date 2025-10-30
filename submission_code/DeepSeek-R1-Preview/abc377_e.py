n, k = map(int, input().split())
p = list(map(int, input().split()))
p = [x - 1 for x in p]  # convert to 0-based

visited = [False] * n
cycle_for = [[] for _ in range(n)]
index_in_cycle = [0] * n

for i in range(n):
    if not visited[i]:
        current = i
        cycle = []
        while True:
            if visited[current]:
                break
            visited[current] = True
            cycle.append(current)
            current = p[current]
        for idx, x in enumerate(cycle):
            cycle_for[x] = cycle
            index_in_cycle[x] = idx

q = [0] * n
for x in range(n):
    cycle = cycle_for[x]
    L = len(cycle)
    e = pow(2, k, L)
    idx = index_in_cycle[x]
    new_idx = (idx + e) % L
    q[x] = cycle[new_idx] + 1  # convert back to 1-based for output

print(' '.join(map(str, q)))