def solve():
    n, m = map(int, input().split())
    trains = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        trains.append((l, d, k, c, a, b))

    def find_f(start_node):
        max_arrival_time = float('-inf')

        def find_paths(current_node, current_time, path):
            nonlocal max_arrival_time
            if current_node == n:
                max_arrival_time = max(max_arrival_time, current_time)
                return

            for l, d, k, c, a, b in trains:
                if a == current_node:
                    for i in range(k):
                        departure_time = l + i * d
                        if departure_time >= current_time:
                            arrival_time = departure_time + c
                            find_paths(b, arrival_time, path + [(departure_time, c, a, b)])

        find_paths(start_node, 0, [])

        if max_arrival_time == float('-inf'):
            return "Unreachable"
        else:
            return str(max_arrival_time)

    for i in range(1, n):
        print(find_f(i))

solve()