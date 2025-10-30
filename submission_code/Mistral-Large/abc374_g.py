import sys
from collections import defaultdict, deque

def min_ng_list_size():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    products = data[1:]

    # Create a graph where each node is a product name and edges represent shared characters
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for product in products:
        a, b = product
        graph[a].append(b)
        in_degree[b] += 1
        out_degree[a] += 1

    # Topological sort using Kahn's algorithm
    queue = deque([node for node in graph if in_degree[node] == 0])
    topo_sorted = []

    while queue:
        node = queue.popleft()
        topo_sorted.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Count the number of strongly connected components
    visited = set()
    scc_count = 0

    def dfs(node):
        stack = [node]
        while stack:
            current = stack[-1]
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
                    break
            else:
                stack.pop()

    for node in topo_sorted:
        if node not in visited:
            visited.add(node)
            dfs(node)
            scc_count += 1

    print(scc_count)

# Read input from stdin, solve the problem, and write output to stdout
if __name__ == "__main__":
    min_ng_list_size()