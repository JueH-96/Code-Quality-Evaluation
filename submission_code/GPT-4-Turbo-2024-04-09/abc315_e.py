import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

dependencies = defaultdict(list)
indegree = [0] * (N + 1)

for i in range(1, N + 1):
    C_i = int(data[index])
    index += 1
    for _ in range(C_i):
        P_ij = int(data[index])
        index += 1
        dependencies[P_ij].append(i)
        indegree[i] += 1

# We need to read book 1, so we start from there
queue = deque()
for i in range(2, N + 1):
    if indegree[i] == 0:
        queue.append(i)

order = []

# Topological sort (Kahn's algorithm)
while queue:
    node = queue.popleft()
    order.append(node)
    for dependent in dependencies[node]:
        indegree[dependent] -= 1
        if indegree[dependent] == 0:
            queue.append(dependent)

# We need to filter the order to only include those necessary to read book 1
necessary_books = set()
queue = deque([1])
while queue:
    node = queue.popleft()
    for pre in dependencies[node]:
        if pre not in necessary_books:
            necessary_books.add(pre)
            queue.append(pre)

# Filter the order list to only include the necessary books
final_order = [book for book in order if book in necessary_books]
print(" ".join(map(str, final_order)))