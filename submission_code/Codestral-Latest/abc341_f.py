import heapq

# Read input
N, M = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]
W = list(map(int, input().split()))
A = list(map(int, input().split()))

# Build adjacency list
adj = [[] for _ in range(N)]
for u, v in edges:
    adj[u-1].append(v-1)
    adj[v-1].append(u-1)

# Initialize heap with vertices having pieces
heap = []
for i in range(N):
    if A[i] > 0:
        heapq.heappush(heap, (-A[i], i))

# Initialize result
result = 0

# Process the heap
while heap:
    count, vertex = heapq.heappop(heap)
    count = -count
    result += count

    # Distribute pieces to adjacent vertices
    total_weight = 0
    for neighbor in adj[vertex]:
        total_weight += W[neighbor]

    if total_weight < W[vertex]:
        for neighbor in adj[vertex]:
            if A[neighbor] > 0:
                heapq.heappush(heap, (-A[neighbor], neighbor))
            A[neighbor] += count
        A[vertex] = 0
    else:
        for neighbor in adj[vertex]:
            if A[neighbor] > 0:
                heapq.heappush(heap, (-A[neighbor], neighbor))
        A[vertex] = 0

# Print the result
print(result)