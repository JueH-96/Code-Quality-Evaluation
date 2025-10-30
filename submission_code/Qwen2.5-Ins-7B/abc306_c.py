# YOUR CODE HERE
from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))

index_map = defaultdict(list)
for i, a in enumerate(A):
    index_map[a].append(i + 1)

result = []
for i in range(1, N + 1):
    result.append(index_map[i][1])

result.sort(key=lambda x: x)

for i in range(len(result) - 1):
    print(result[i], end=' ')
print(result[-1])