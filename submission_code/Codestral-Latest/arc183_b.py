import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1

def bfs(start, target, graph, N):
    queue = deque([start])
    visited = set()
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

def can_transform(A, B, N, K):
    graph = defaultdict(list)
    for i in range(N):
        for j in range(max(0, i-K), min(N, i+K+1)):
            graph[A[i]].append(A[j])

    for i in range(N):
        if not bfs(A[i], B[i], graph, N):
            return False
    return True

results = []
for _ in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    A = list(map(int, data[index + 2:index + 2 + N]))
    B = list(map(int, data[index + 2 + N:index + 2 + 2*N]))
    index += 2 + 2*N

    if can_transform(A, B, N, K):
        results.append("Yes")
    else:
        results.append("No")

print("
".join(results))