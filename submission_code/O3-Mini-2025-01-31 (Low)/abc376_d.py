def main():
    import sys
    from collections import deque
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the graph. Also, mark which vertices have an edge back to vertex 1.
    graph = [[] for _ in range(n + 1)]
    back_to_1 = [False] * (n + 1)
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        if v == 1:
            back_to_1[u] = True  # This vertex u has an edge to 1
    
    # Use a BFS from vertex 1 to compute the shortest distance from 1 to every other vertex.
    INF = 10**9
    distance = [INF] * (n + 1)
    distance[1] = 0
    dq = deque([1])
    
    while dq:
        current = dq.popleft()
        for neighbor in graph[current]:
            if distance[neighbor] > distance[current] + 1:
                distance[neighbor] = distance[current] + 1
                dq.append(neighbor)
    
    # Now the cycle that contains vertex 1 would be a path from 1 to some vertex v and then an edge from v back to 1.
    # So, the candidate cycle length is distance[v] + 1 for some v with an edge to 1.
    ans = INF
    # We consider vertices v != 1 since self-loop is not allowed by the problem constraints.
    for v in range(2, n + 1):
        if back_to_1[v] and distance[v] != INF:
            ans = min(ans, distance[v] + 1)
    
    if ans == INF:
        print(-1)
    else:
        print(ans)
        
if __name__ == '__main__':
    main()