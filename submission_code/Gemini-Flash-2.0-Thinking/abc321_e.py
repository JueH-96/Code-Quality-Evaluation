def solve():
    n, x, k = map(int, input().split())

    def get_neighbors(node):
        neighbors = []
        parent = node // 2
        if parent > 0:
            neighbors.append(parent)
        child1 = 2 * node
        if child1 <= n:
            neighbors.append(child1)
        child2 = 2 * node + 1
        if child2 <= n:
            neighbors.append(child2)
        return neighbors

    def bfs(start_node):
        distances = {}
        queue = [(start_node, 0)]
        distances[start_node] = 0
        visited = {start_node}

        while queue:
            current_node, current_dist = queue.pop(0)

            for neighbor in get_neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))
        return distances

    distances_from_x = bfs(x)
    count = 0
    for i in range(1, n + 1):
        if i in distances_from_x and distances_from_x[i] == k:
            count += 1
    return count

t = int(input())
for _ in range(t):
    result = solve()
    print(result)