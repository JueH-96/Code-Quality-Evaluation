import sys
input = sys.stdin.read

def count_reachable_pairs(N, edges):
    # Initialize the graph
    graph = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        graph[i].append(edges[i-1])

    # Initialize variables to count reachable pairs
    visited = [False] * (N + 1)
    reachable_count = [0] * (N + 1)
    total_pairs = 0

    # Function to perform DFS and count reachable nodes
    def dfs(u):
        stack = [u]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                reachable_count[u] += 1
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)

    # Perform DFS from each node
    for i in range(1, N+1):
        if not visited[i]:
            dfs(i)
            total_pairs += reachable_count[i]

    return total_pairs

# Read input
data = input().strip().split()
N = int(data[0])
edges = list(map(int, data[1:]))

# Calculate and print the result
result = count_reachable_pairs(N, edges)
print(result)