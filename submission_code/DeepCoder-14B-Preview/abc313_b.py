n, m = map(int, input().split())
in_degree = [0] * (n + 1)

for _ in range(m):
    a, b = map(int, input().split())
    in_degree[b] += 1

candidates = [i for i in range(1, n + 1) if in_degree[i] == 0]

if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)