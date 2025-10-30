import sys
from collections import defaultdict, deque

def is_good_graph(N, edges, K, pairs):
    def bfs(start, target):
        queue = deque([start])
        visited = set([start])
        while queue:
            node = queue.popleft()
            if node == target:
                return True
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    for x, y in pairs:
        if bfs(x, y):
            return "No"
    return "Yes"

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
edges = [(int(data[2 + 2 * i]) - 1, int(data[3 + 2 * i]) - 1) for i in range(M)]
K = int(data[2 + 2 * M])
pairs = [(int(data[3 + 2 * M + 2 * i]) - 1, int(data[4 + 2 * M + 2 * i]) - 1) for i in range(K)]
Q = int(data[3 + 2 * M + 2 * K])
questions = [(int(data[4 + 2 * M + 2 * K + 2 * i]) - 1, int(data[5 + 2 * M + 2 * K + 2 * i]) - 1) for i in range(Q)]

# Process each question
results = [is_good_graph(N, edges, K, pairs + [q]) for q in questions]

# Output results
for result in results:
    print(result)