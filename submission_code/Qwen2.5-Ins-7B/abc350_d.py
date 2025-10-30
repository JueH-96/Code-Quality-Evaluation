# YOUR CODE HERE
from collections import defaultdict

n, m = map(int, input().split())
friendship_count = defaultdict(int)
degree = [0] * (n + 1)

for _ in range(m):
    x, y = map(int, input().split())
    friendship_count[x] += 1
    friendship_count[y] += 1
    degree[x] += 1
    degree[y] += 1

result = 0
for x in range(1, n + 1):
    for y in range(x + 1, n + 1):
        if degree[x] + degree[y] <= n and friendship_count[x] and friendship_count[y]:
            result += min(friendship_count[x], friendship_count[y], n - degree[x] - degree[y])

print(result)