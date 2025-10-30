def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))

    def shortest_distance(a, b, closed_bridge=None):
        if closed_bridge is None:
            return min(abs(a - b), n - abs(a - b))
        else:
            u, v = closed_bridge
            if u > v:
                u, v = v, u

            if u != v + 1 and not (u == 1 and v == n):
                raise ValueError("Invalid bridge")

            if u == v + 1:
                if (min(a, b) <= v and max(a, b) > v):
                    return abs(a - b)
                else:
                    return min(abs(a - b), n - abs(a - b))
            else: # u == 1 and v == n
                if (a == 1 and b == n) or (a == n and b == 1):
                    return n - 1
                else:
                    return min(abs(a - b), n - abs(a - b))

    min_tour_length = float('inf')

    # Case: No bridge is closed (for debugging/understanding)
    # current_length = 0
    # for i in range(m - 1):
    #     current_length += shortest_distance(x[i], x[i+1])
    # print(f"No bridge closed: {current_length}")

    for closed_bridge_index in range(n):
        closed_bridge = None
        if closed_bridge_index < n - 1:
            closed_bridge = (closed_bridge_index + 1, closed_bridge_index + 2)
        else:
            closed_bridge = (n, 1)

        current_tour_length = 0
        for i in range(m - 1):
            dist = 0
            start_node = x[i]
            end_node = x[i+1]

            q = [(start_node, 0, [start_node])]
            visited = {start_node}
            found = False

            while q:
                curr, d, path = q.pop(0)

                if curr == end_node:
                    dist = d
                    found = True
                    break

                neighbors = []
                if closed_bridge is None or (min(curr, curr + 1 if curr < n else 1) != min(closed_bridge[0], closed_bridge[1]) or max(curr, curr + 1 if curr < n else 1) != max(closed_bridge[0], closed_bridge[1])):
                    neighbors.append(curr + 1 if curr < n else 1)
                if closed_bridge is None or (min(curr, curr - 1 if curr > 1 else n) != min(closed_bridge[0], closed_bridge[1]) or max(curr, curr - 1 if curr > 1 else n) != max(closed_bridge[0], closed_bridge[1])):
                    neighbors.append(curr - 1 if curr > 1 else n)

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1, path + [neighbor]))

            if not found:
                raise Exception("Should always find a path")
            current_tour_length += dist

        min_tour_length = min(min_tour_length, current_tour_length)

    print(min_tour_length)

solve()