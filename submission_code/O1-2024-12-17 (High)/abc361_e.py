def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])

    # Build adjacency list
    graph = [[] for _ in range(N+1)]
    sum_of_edges = 0
    idx = 1
    for _ in range(N-1):
        A = int(input_data[idx])
        B = int(input_data[idx+1])
        C = int(input_data[idx+2])
        idx += 3
        graph[A].append((B, C))
        graph[B].append((A, C))
        sum_of_edges += C

    # Function to find the farthest node and the distance to it
    # from a given start node in a tree.
    def find_farthest_node(start):
        dist = [-1] * (N+1)
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt, cost in graph[node]:
                if dist[nxt] == -1:
                    dist[nxt] = dist[node] + cost
                    queue.append(nxt)
        farthest = start
        for i in range(1, N+1):
            if dist[i] > dist[farthest]:
                farthest = i
        return farthest, dist[farthest]

    # 1. Find a farthest node x from an arbitrary node (e.g., 1).
    x, _ = find_farthest_node(1)
    # 2. From x, find the farthest node y, and the distance to it is the diameter.
    y, diameter = find_farthest_node(x)

    # The minimum route that visits all nodes and does NOT return to start:
    # 2 * (sum of all edge lengths) - diameter
    answer = 2 * sum_of_edges - diameter
    print(answer)

# Do not forget to call main.
if __name__ == "__main__":
    main()