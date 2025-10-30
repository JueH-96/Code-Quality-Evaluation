# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

queries = data[2:]

# Adjacency list to store the graph
graph = [[] for _ in range(N + 1)]

output = []

index = 0
for _ in range(Q):
    query_type = int(queries[index])
    if query_type == 1:
        u = int(queries[index + 1])
        v = int(queries[index + 2])
        graph[u].append(v)
        graph[v].append(u)
        index += 3
    elif query_type == 2:
        v = int(queries[index + 1])
        k = int(queries[index + 2])
        if len(graph[v]) < k:
            output.append("-1")
        else:
            # Use a max-heap to find the k-th largest element
            largest_k = heapq.nlargest(k, graph[v])
            output.append(str(largest_k[-1]))
        index += 3

print("
".join(output))