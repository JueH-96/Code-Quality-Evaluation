def main():
    import sys
    from collections import deque
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Read inputs
    n = int(input_data[0])
    D = int(input_data[1])
    pts = []
    index = 2
    for _ in range(n):
        x = int(input_data[index])
        y = int(input_data[index+1])
        pts.append((x, y))
        index += 2

    # Precompute squared distance threshold.
    D_sq = D * D

    # Build the graph: each person is a node and there is an edge between two persons if they're within distance D.
    graph = [[] for _ in range(n)]
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy <= D_sq:
                graph[i].append(j)
                graph[j].append(i)

    # Use BFS starting from person 1 (index 0) to mark all infected individuals.
    infected = [False] * n
    queue = deque([0])
    infected[0] = True
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if not infected[neighbor]:
                infected[neighbor] = True
                queue.append(neighbor)

    # Output the result for each person.
    output = []
    for i in range(n):
        output.append("Yes" if infected[i] else "No")
    print("
".join(output))

main()