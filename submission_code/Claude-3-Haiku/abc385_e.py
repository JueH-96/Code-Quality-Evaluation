import sys

def find_min_vertices_to_delete(N, edges):
    # Create adjacency list
    adj_list = [[] for _ in range(N)]
    for u, v in edges:
        adj_list[u-1].append(v-1)
        adj_list[v-1].append(u-1)

    # DFS to find the Snowflake Tree
    visited = [False] * N
    def dfs(node, parent, x, y):
        visited[node] = True
        if len(adj_list[node]) == x + 1:
            # This is a vertex in the Snowflake Tree
            for neighbor in adj_list[node]:
                if neighbor != parent and not visited[neighbor]:
                    dfs(neighbor, node, y, y)
            return 0
        else:
            # This is not a vertex in the Snowflake Tree
            return 1

    total_deleted = 0
    for i in range(N):
        if not visited[i]:
            total_deleted += dfs(i, -1, 0, 0)

    return total_deleted

# Read input
N = int(input())
edges = []
for _ in range(N-1):
    u, v = map(int, input().split())
    edges.append((u, v))

# Solve the problem
answer = find_min_vertices_to_delete(N, edges)
print(answer)