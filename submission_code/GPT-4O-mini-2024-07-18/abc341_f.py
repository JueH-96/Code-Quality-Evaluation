def max_operations(N, M, edges, W, A):
    from collections import defaultdict, deque

    # Create adjacency list for the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    total_operations = 0
    queue = deque()

    # Initialize the queue with all vertices that have pieces
    for i in range(1, N + 1):
        if A[i - 1] > 0:
            queue.append(i)

    while queue:
        x = queue.popleft()
        while A[x - 1] > 0:
            A[x - 1] -= 1
            total_operations += 1
            
            # Try to distribute pieces to adjacent vertices
            S = []
            for neighbor in graph[x]:
                if W[neighbor - 1] < W[x - 1]:
                    S.append(neighbor)

            # Place one piece on each vertex in S
            for neighbor in S:
                A[neighbor - 1] += 1
                if A[neighbor - 1] == 1:  # If it just received a piece, add to queue
                    queue.append(neighbor)

    return total_operations

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
edges = [tuple(map(int, line.split())) for line in data[1:M + 1]]
W = list(map(int, data[M + 1].split()))
A = list(map(int, data[M + 2].split()))

result = max_operations(N, M, edges, W, A)
print(result)