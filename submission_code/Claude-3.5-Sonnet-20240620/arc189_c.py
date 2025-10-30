# YOUR CODE HERE
from collections import deque

def solve():
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))

    # Create a graph where each box is a node
    graph = [[] for _ in range(N)]
    for i in range(N):
        if A[i] == 1:
            graph[i].append(P[i] - 1)
        if B[i] == 1:
            graph[i].append(Q[i] - 1)

    # BFS to find the minimum number of operations
    queue = deque([(i, 0) for i in range(N) if i != X - 1 and (A[i] == 1 or B[i] == 1)])
    visited = set([i for i in range(N) if i != X - 1 and A[i] == 0 and B[i] == 0])

    while queue:
        node, dist = queue.popleft()
        if node == X - 1:
            continue
        if node in visited:
            continue
        visited.add(node)
        for next_node in graph[node]:
            if next_node not in visited:
                queue.append((next_node, dist + 1))

    if len(visited) == N - 1:
        print(len([i for i in range(N) if i != X - 1 and (A[i] == 1 or B[i] == 1)]))
    else:
        print(-1)

solve()