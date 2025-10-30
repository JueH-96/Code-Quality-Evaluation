def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return

    # Read N and prepare graph
    it = iter(input_data)
    n = int(next(it))
    graph = [[] for _ in range(n + 1)]
    total_distance = 0

    # Build the tree graph and sum all road lengths.
    for _ in range(n - 1):
        a = int(next(it))
        b = int(next(it))
        c = int(next(it))
        graph[a].append((b, c))
        graph[b].append((a, c))
        total_distance += c

    # Iterative DFS to compute farthest node and the distance from a given start.
    def dfs(start):
        stack = [(start, -1, 0)]
        farthest_node = start
        max_dist = 0
        while stack:
            node, parent, dist = stack.pop()
            if dist > max_dist:
                max_dist = dist
                farthest_node = node
            for neighbor, weight in graph[node]:
                if neighbor == parent:
                    continue
                stack.append((neighbor, node, dist + weight))
        return farthest_node, max_dist

    # First DFS from an arbitrary node (here, node 1) to find a farthest node.
    far_node, _ = dfs(1)
    # Second DFS from that farthest node to determine the diameter (max distance).
    _, diameter = dfs(far_node)

    # The minimal travel distance is twice the sum of all road lengths minus the diameter.
    result = 2 * total_distance - diameter

    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()