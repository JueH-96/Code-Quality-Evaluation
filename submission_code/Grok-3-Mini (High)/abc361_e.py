import sys
import collections

def bfs(graph, start, N):
    dist = [-1] * (N + 1)
    dist[start] = 0
    queue = collections.deque()
    queue.append(start)
    while queue:
        node = queue.popleft()
        for nbr, weight in graph[node]:
            if dist[nbr] == -1:
                dist[nbr] = dist[node] + weight
                queue.append(nbr)
    return dist

# Read input
data = list(map(int, sys.stdin.read().split()))
N = data[0]
graph = [[] for _ in range(N + 1)]
total_sum = 0

# Build graph and compute sum of all edge weights
for i in range(N - 1):
    A = data[1 + 3 * i]
    B = data[2 + 3 * i]
    C = data[3 + 3 * i]
    total_sum += C
    graph[A].append((B, C))
    graph[B].append((A, C))

# Find diameter of the tree
# Step 1: BFS from node 1 to find a farthest node A
dist1 = bfs(graph, 1, N)
A = max(range(1, N + 1), key=lambda x: dist1[x])

# Step 2: BFS from node A to find the farthest node and the diameter length
distA = bfs(graph, A, N)
diam_length = max(distA[1:])

# Calculate the minimum travel distance
answer = 2 * total_sum - diam_length

# Output the answer
print(answer)