def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the directed graph as an adjacency list.
    # We use 1-indexing, so our list has n+1 entries.
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
    
    # We'll perform a BFS starting from vertex 1 to find the shortest path
    # from vertex 1 to every other vertex.
    # The idea: any cycle containing vertex 1 must eventually come back to 1
    # via some edge from some vertex u. Thus, if we discover an edge u -> 1,
    # then the cycle length is dist[u] + 1.
    dist = [-1] * (n + 1)
    dist[1] = 0
    dq = deque([1])
    
    best = float('inf')
    
    while dq:
        u = dq.popleft()
        for v in graph[u]:
            if v == 1:
                # We have a cycle: 1 -> ... -> u -> 1.
                best = min(best, dist[u] + 1)
                # Do not add 1 into the queue since we've already used it as the start.
            elif dist[v] == -1:
                dist[v] = dist[u] + 1
                dq.append(v)
    
    # Output the answer based on whether a cycle was found.
    if best == float('inf'):
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(best))

# Call the main function.
main()