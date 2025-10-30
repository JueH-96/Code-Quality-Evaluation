def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])

    # Adjacency list for the forward graph
    graph = [[] for _ in range(N+1)]
    # Adjacency list for the reversed graph
    rgraph = [[] for _ in range(N+1)]
    edges = []

    idx = 2
    for _ in range(M):
        a = int(input_data[idx])
        b = int(input_data[idx+1])
        idx += 2
        graph[a].append(b)
        rgraph[b].append(a)
        edges.append((a, b))

    # BFS from 1 in the original graph
    dist1 = [-1] * (N+1)
    dist1[1] = 0
    queue = deque([1])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            if dist1[nxt] == -1:
                dist1[nxt] = dist1[cur] + 1
                queue.append(nxt)

    # BFS from 1 in the reversed graph
    dist2 = [-1] * (N+1)
    dist2[1] = 0
    queue = deque([1])
    while queue:
        cur = queue.popleft()
        for nxt in rgraph[cur]:
            if dist2[nxt] == -1:
                dist2[nxt] = dist2[cur] + 1
                queue.append(nxt)

    # Find the minimum cycle that contains vertex 1
    answer = float('inf')
    for u, v in edges:
        if dist1[u] != -1 and dist2[v] != -1:
            cycle_len = dist1[u] + 1 + dist2[v]
            if cycle_len < answer:
                answer = cycle_len

    if answer == float('inf'):
        print(-1)
    else:
        print(answer)

# Do not forget to call main!
if __name__ == "__main__":
    main()