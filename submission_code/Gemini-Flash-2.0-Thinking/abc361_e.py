import collections

def solve():
    n = int(input())
    adj = collections.defaultdict(list)
    total_edge_length = 0
    for _ in range(n - 1):
        u, v, c = map(int, input().split())
        adj[u].append((v, c))
        adj[v].append((u, c))
        total_edge_length += c

    if n == 2:
        print(total_edge_length)
        return

    def get_farthest_node(start_node):
        distances = [-1] * (n + 1)
        distances[start_node] = 0
        queue = collections.deque([(start_node, 0)])
        farthest_node = start_node
        max_distance = 0

        while queue:
            curr, dist = queue.popleft()
            if dist > max_distance:
                max_distance = dist
                farthest_node = curr

            for neighbor, weight in adj[curr]:
                if distances[neighbor] == -1:
                    distances[neighbor] = dist + weight
                    queue.append((neighbor, dist + weight))
        return farthest_node, max_distance

    farthest1, _ = get_farthest_node(1)
    _, longest_path_length = get_farthest_node(farthest1)

    min_travel_distance = 2 * total_edge_length - longest_path_length
    print(min_travel_distance)

solve()