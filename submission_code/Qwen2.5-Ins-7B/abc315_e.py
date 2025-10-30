# YOUR CODE HERE
from collections import deque, defaultdict

N = int(input())
graph = defaultdict(list)
indegree = [0] * (N + 1)
books = [[] for _ in range(N + 1)]

for i in range(1, N + 1):
    C, *P = map(int, input().split())
    books[i] = P
    for p in P:
        graph[p].append(i)
        indegree[i] += 1

queue = deque([i for i in range(1, N + 1) if indegree[i] == 0])
result = []

while queue:
    current = queue.popleft()
    result.append(current)
    for next_book in graph[current]:
        indegree[next_book] -= 1
        if indegree[next_book] == 0:
            queue.append(next_book)

print(*[book for book in result if book != 1])