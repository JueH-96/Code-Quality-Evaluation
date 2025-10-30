import sys
from collections import defaultdict, deque

def main():
    N = int(sys.stdin.readline())
    edges = []
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    nodes = set()
    self_loops = set()

    for _ in range(N):
        s = sys.stdin.readline().strip()
        a, b = s[0], s[1]
        edges.append((a, b))
        out_degree[a] += 1
        in_degree[b] += 1
        nodes.add(a)
        nodes.add(b)
        if a == b:
            self_loops.add(a)

    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b]  # Ensure even nodes with no outgoing edges are present

    visited = {}
    for node in nodes:
        visited[node] = 0

    components = []
    for node in nodes:
        if visited[node] == 0:
            q = deque()
            q.append(node)
            visited[node] = 1
            component = set()
            component.add(node)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if visited[v] == 0:
                        visited[v] = 1
                        component.add(v)
                        q.append(v)
                for v in adj:
                    if u in adj[v]:
                        if visited[v] == 0:
                            visited[v] = 1
                            component.add(v)
                            q.append(v)
            components.append(component)

    def is_strongly_connected(subgraph):
        if len(subgraph) == 1:
            return True
        for root in subgraph:
            break
        visited_strong = {u: 0 for u in subgraph}
        visited_strong[root] = 1
        q = deque([root])
        count = 1
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in subgraph and visited_strong[v] == 0:
                    visited_strong[v] = 1
                    count += 1
                    q.append(v)
        return count == len(subgraph)

    total = 0
    for comp in components:
        sum_out_in = 0
        all_degree_1 = True
        for node in comp:
            if out_degree[node] != in_degree[node] or out_degree[node] != 1:
                all_degree_1 = False
            diff = out_degree[node] - in_degree[node]
            if diff > 0:
                sum_out_in += diff
        if all_degree_1 and is_strongly_connected(comp):
            total += 1
        else:
            total += sum_out_in

    for node in self_loops:
        found = False
        for comp in components:
            if node in comp:
                found = True
                break
        if not found:
            total += 1

    print(total)

if __name__ == '__main__':
    main()