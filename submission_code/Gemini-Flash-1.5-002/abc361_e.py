# YOUR CODE HERE
import itertools

def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
        edges.append((v, u, w))

    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges:
        graph[u].append((v, w))

    min_dist = float('inf')
    for start_node in range(1, n + 1):
        for permutation in itertools.permutations(range(1, n + 1)):
            if permutation[0] != start_node:
                continue
            
            total_dist = 0
            current_node = start_node
            visited = {start_node}
            
            for next_node in permutation[1:]:
                q = [(current_node, 0)]
                visited_bfs = {current_node}
                found = False
                while q:
                    curr, dist = q.pop(0)
                    if curr == next_node:
                        total_dist += dist
                        current_node = next_node
                        visited.add(next_node)
                        found = True
                        break
                    for neighbor, weight in graph[curr]:
                        if neighbor not in visited_bfs:
                            q.append((neighbor, dist + weight))
                            visited_bfs.add(neighbor)
                if not found:
                    total_dist = float('inf')
                    break

            min_dist = min(min_dist, total_dist)

    print(min_dist)

solve()