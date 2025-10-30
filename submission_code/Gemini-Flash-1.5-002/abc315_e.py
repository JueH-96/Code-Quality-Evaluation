# YOUR CODE HERE
n = int(input())
reqs = []
for i in range(n):
    line = list(map(int, input().split()))
    reqs.append(line[1:])

graph = {i: [] for i in range(1, n + 1)}
indegree = {i: 0 for i in range(1, n + 1)}
for i in range(n):
    for req in reqs[i]:
        graph[req].append(i + 1)
        indegree[i + 1] += 1

queue = []
for i in range(1, n + 1):
    if indegree[i] == 0:
        queue.append(i)

result = []
while queue:
    curr = queue.pop(0)
    if curr == 1:
        break
    result.append(curr)
    for neighbor in graph[curr]:
        indegree[neighbor] -= 1
        if indegree[neighbor] == 0:
            queue.append(neighbor)

print(*result[::-1])