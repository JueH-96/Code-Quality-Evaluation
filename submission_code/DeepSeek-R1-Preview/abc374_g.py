import sys
from collections import defaultdict, deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        edges.append((s[0], s[1]))

    # Build in_degree and out_degree
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    adj = defaultdict(list)
    nodes = set()
    for a, b in edges:
        out_degree[a] += 1
        in_degree[b] += 1
        adj[a].append(b)
        adj[b].append(a)  # for undirected connectivity
        nodes.add(a)
        nodes.add(b)

    # Find connected components using BFS (undirected)
    visited = set()
    components = []
    for node in nodes:
        if node not in visited:
            q = deque()
            q.append(node)
            visited.add(node)
            component = set()
            component.add(node)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        component.add(v)
                        q.append(v)
            components.append(component)

    total_trails = 0

    for comp in components:
        sum_positive = 0
        sum_delta = 0
        has_edges = False
        # Check if the component has any edges
        for a, b in edges:
            if a in comp or b in comp:
                has_edges = True
                break
        if not has_edges:
            continue  # no edges, no trails needed

        for node in comp:
            out_d = out_degree.get(node, 0)
            in_d = in_degree.get(node, 0)
            delta = out_d - in_d
            sum_positive += max(0, delta)
            sum_delta += delta

        if sum_positive == 0:
            # Check if component has any edges
            total_trails += 1
        else:
            total_trails += sum_positive

    print(total_trails)

if __name__ == '__main__':
    main()