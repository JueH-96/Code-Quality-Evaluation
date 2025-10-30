def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    m = int(next(it))

    # Build an undirected graph with weighted edges.
    # For every piece of information, person A's view gives:
    #   coordinate[B] = coordinate[A] + (X, Y)
    # Thus we add an edge from A to B with weight (X,Y) and reverse edge from B to A with (-X,-Y)
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        dx = int(next(it))
        dy = int(next(it))
        graph[a].append((b, dx, dy))
        graph[b].append((a, -dx, -dy))

    # Person 1 is fixed at (0, 0)
    coords = [None] * (n + 1)
    coords[1] = (0, 0)

    # Use BFS to determine coordinates for all people in the connected component of person 1.
    # If a person is not visited, then their component has translation freedom and they are undecidable.
    q = deque([1])
    while q:
        curr = q.popleft()
        curr_x, curr_y = coords[curr]
        for neighbor, dx, dy in graph[curr]:
            new_x = curr_x + dx
            new_y = curr_y + dy
            if coords[neighbor] is None:
                coords[neighbor] = (new_x, new_y)
                q.append(neighbor)
            # Since the input is consistent, we do not need to verify if an already determined coordinate matches new_x, new_y.

    # Produce output.
    # For each person, if their coordinates are determined, output them.
    # Otherwise, print "undecidable".
    output_lines = []
    for i in range(1, n + 1):
        if coords[i] is None:
            output_lines.append("undecidable")
        else:
            output_lines.append(f"{coords[i][0]} {coords[i][1]}")

    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()