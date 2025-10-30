import sys
from collections import defaultdict, deque

def main():
    N = int(sys.stdin.readline())
    edges = []
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    nodes = set()

    for _ in range(N):
        s = sys.stdin.readline().strip()
        u, v = s[0], s[1]
        edges.append((u, v))
        out_degree[u] += 1
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)

    # Build undirected graph to find connected components
    from collections import defaultdict, deque
    undirected_graph = defaultdict(list)
    for u, v in edges:
        undirected_graph[u].append(v)
        undirected_graph[v].append(u)

    visited = {}
    for node in nodes:
        visited[node] = False

    # Function to find connected components
    def find_components():
        components = []
        for node in nodes:
            if not visited[node]:
                q = deque()
                q.append(node)
                visited[node] = True
                component = []
                while q:
                    current = q.popleft()
                    component.append(current)
                    for neighbor in undirected_graph[current]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            q.append(neighbor)
                components.append(component)
        return components

    components = find_components()

    sum_degree_diff = 0
    extra = 0

    # For each component, calculate sum_degree_diff and extra
    for comp in components:
        comp_has_edge = False
        comp_sum_diff = 0
        comp_sum_degree_diff = 0
        for node in comp:
            od = out_degree.get(node, 0)
            id_ = in_degree.get(node, 0)
            if od > 0 or id_ > 0:
                comp_has_edge = True
            diff = od - id_
            comp_sum_diff += diff
            comp_sum_degree_diff += max(0, od - id_)
        if comp_has_edge:
            if comp_sum_degree_diff == 0:
                extra += 1
    sum_degree_diff = sum(max(0, out_degree.get(node, 0) - in_degree.get(node, 0)) for node in nodes)

    result = sum_degree_diff + extra
    print(result)

if __name__ == "__main__":
    main()