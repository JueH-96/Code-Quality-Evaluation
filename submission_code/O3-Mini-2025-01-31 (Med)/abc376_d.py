def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the graph as an adjacency list.
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        graph[a].append(b)
    
    # We want to find a cycle that contains vertex 1 (i.e. a route starting at 1 and eventually returning to 1).
    # We can perform a Breadth-First Search (BFS) starting from vertex 1.
    # Whenever we find an edge from a vertex v to 1, a cycle is confirmed.
    # The cycle length is dist[v] + 1 (the edge from v to 1).
    # As BFS processes vertices in increasing order of distance, the first discovered cycle will have the minimal number of edges.
    
    # Setup BFS.
    dist = [-1] * (n + 1)
    dist[1] = 0
    q = deque([1])
    answer = -1

    while q:
        v = q.popleft()
        for w in graph[v]:
            if w == 1:
                # Found a cycle that returns to 1.
                cycle_length = dist[v] + 1
                # Since BFS ensures increasing order of distance,
                # the first time we encounter an edge to 1, it is the shortest cycle.
                if answer == -1 or cycle_length < answer:
                    answer = cycle_length
            else:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
    
    print(answer)


if __name__ == '__main__':
    main()