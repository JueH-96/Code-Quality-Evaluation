from collections import defaultdict
import heapq

# Read the number of vertices N and the number of queries Q
N, Q = map(int, input().split())

# Initialize a dictionary to store the adjacency list
graph = defaultdict(list)

# Process each query
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1 query: Add an edge between u and v
        _, u, v = query
        heapq.heappush(graph[u], -v)  # Use a min heap with negative values to simulate max heap
        heapq.heappush(graph[v], -u)
    elif query[0] == 2:
        # Type 2 query: Print the k-th largest vertex number connected to v
        _, v, k = query
        if len(graph[v]) < k:
            print(-1)
        else:
            # Create a copy of the heap to avoid modifying the original
            temp_heap = graph[v][:]
            kth_largest = None
            for _ in range(k):
                kth_largest = -heapq.heappop(temp_heap)
            print(kth_largest)