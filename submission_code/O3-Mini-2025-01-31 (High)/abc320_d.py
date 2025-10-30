def main():
    import sys
    from collections import deque

    data = sys.stdin.buffer.read().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Create graph for relative placements.
    # For each piece of information (a, b, x, y):
    #   pos(b) = pos(a) + (x, y)
    # So we add an edge from a to b with delta (x,y) and
    # an edge from b to a with delta (-x,-y)
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        x = int(next(it))
        y = int(next(it))
        graph[a].append((b, x, y))
        graph[b].append((a, -x, -y))
    
    # positions[i] will store the unique coordinate of person i if it's determined,
    # otherwise it will stay as None.
    # Person 1 is fixed at (0, 0).
    positions = [None] * (N + 1)
    positions[1] = (0, 0)

    # Use BFS to assign positions for all persons that are connected to person 1.
    dq = deque([1])
    while dq:
        current = dq.popleft()
        cur_x, cur_y = positions[current]
        for neighbor, dx, dy in graph[current]:
            new_x = cur_x + dx
            new_y = cur_y + dy
            if positions[neighbor] is None:
                positions[neighbor] = (new_x, new_y)
                dq.append(neighbor)
            else:
                # If already assigned, the new computed coordinate should match the old one.
                # The problem guarantees consistency, so this check is optional.
                if positions[neighbor] != (new_x, new_y):
                    # Inconsistent information, but per problem statement this never happens.
                    pass

    # For each person, if the position is not determined (i.e., not connected to person 1),
    # output "undecidable". Otherwise, output the position.
    output = []
    for i in range(1, N + 1):
        if positions[i] is None:
            output.append("undecidable")
        else:
            output.append(f"{positions[i][0]} {positions[i][1]}")
    
    sys.stdout.write("
".join(output))


if __name__ == "__main__":
    main()