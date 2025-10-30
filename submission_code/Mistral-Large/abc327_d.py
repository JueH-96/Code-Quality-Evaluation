import sys
input = sys.stdin.read

def is_good_pair(N, M, A, B):
    # Create a graph where each number is a node
    # and each pair (A_i, B_i) forms an edge.
    from collections import defaultdict

    graph = defaultdict(list)
    for a, b in zip(A, B):
        graph[a].append(b)
        graph[b].append(a)

    # Check if the graph is bipartite using BFS
    from collections import deque

    color = {}

    def bfs(start):
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
        return True

    for i in range(1, N + 1):
        if i not in color:
            if not bfs(i):
                return "No"

    return "Yes"

data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+M]))
B = list(map(int, data[2+M:2+M+M]))

print(is_good_pair(N, M, A, B))