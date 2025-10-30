n, m = map(int, input().split())
in_degree = [0] * (n + 1)

for _ in range(m):
    a, b = map(int, input().split())
    in_degree[b] += 1

sources = [i for i in range(1, n + 1) if in_degree[i] == 0]

if len(sources) == 1:
    print(sources[0])
else:
    print(-1)