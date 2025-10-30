def solve():
    n, m = map(int, input().split())
    trains_data = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        trains_data.append((l, d, k, c, a, b))

    trains = []
    for l, d, k, c, a, b in trains_data:
        for i in range(k):
            departure_time = l + i * d
            arrival_time = departure_time + c
            trains.append((departure_time, arrival_time, a, b))

    f = [-float('inf')] * (n + 1)
    f[n] = 0

    changed = True
    while changed:
        changed = False
        for dep_time, arr_time, start_station, end_station in trains:
            if f[end_station] != -float('inf'):
                f[start_station] = max(f[start_station], f[end_station] - (arr_time - dep_time))
                if f[start_station] > -float('inf'):
                    changed = True

    results = []
    for start_node in range(1, n):
        max_arrival_time = -float('inf')
        for start_time in range(101): # Iterate over possible start times
            q = [(start_time, start_node)]
            max_arrival = -float('inf')

            while q:
                current_time, current_station = q.pop(0)

                if current_station == n:
                    max_arrival = max(max_arrival, current_time)
                    continue

                for t_dep, t_arr, from_station, to_station in trains:
                    if from_station == current_station and t_dep >= current_time:
                        q.append((t_arr, to_station))

            max_arrival_time = max(max_arrival_time, max_arrival)

        def get_f_s(start_s):
            max_arrival_time_s = -float('inf')

            reachable_arrivals = {}

            def find_paths(current_station, current_time):
                if current_station == n:
                    reachable_arrivals[start_s] = max(reachable_arrivals.get(start_s, -float('inf')), current_time)
                    return

                for t_dep, t_arr, from_station, to_station in trains:
                    if from_station == current_station and t_dep >= current_time:
                        find_paths(to_station, t_arr)

            for start_t in range(101):
                find_paths(start_s, start_t)
            return reachable_arrivals.get(start_s, -float('inf'))

        # results.append(get_f_s(start_node))

        max_arrival_time = -float('inf')

        q = [(0, start_node, [])] # (current_time, current_station, path)

        while q:
            current_time, current_station, path = q.pop(0)

            if current_station == n:
                max_arrival_time = max(max_arrival_time, current_time)
                continue

            for t_dep, t_arr, from_station, to_station in trains:
                if from_station == current_station and t_dep >= current_time:
                    q.append((t_arr, to_station, path + [(t_dep, t_arr, from_station, to_station)]))

        results.append(max_arrival_time if max_arrival_time != -float('inf') else "Unreachable")

    f_values = [-float('inf')] * (n + 1)

    for start_node in range(1, n):
        max_f_s = -float('inf')
        for initial_time in range(200):
            q = [(initial_time, start_node, [])]
            while q:
                current_time, current_station, path = q.pop(0)

                if current_station == n:
                    max_f_s = max(max_f_s, current_time)
                    continue

                for t_dep, t_arr, from_station, to_station in trains:
                    if from_station == current_station and t_dep >= current_time:
                        q.append((t_arr, to_station, path + [(t_dep, t_arr, from_station, to_station)]))
        f_values[start_node] = max_f_s

    for i in range(1, n):
        max_arrival = -float('inf')
        for t in range(200):
            q = [(t, i)]
            visited = set()
            while q:
                current_t, current_s = q.pop(0)
                if (current_t, current_s) in visited:
                    continue
                visited.add((current_t, current_s))

                if current_s == n:
                    max_arrival = max(max_arrival, current_t)
                    continue

                for t_dep, t_arr, from_node, to_node in trains:
                    if from_node == current_s and t_dep >= current_t:
                        q.append((t_arr, to_node))
        results.append(max_arrival if max_arrival > -float('inf') else "Unreachable")

    dp = [-float('inf')] * (n + 1)
    dp[n] = 0

    for _ in range(n + 1):
        updated = False
        for t_dep, t_arr, u, v in trains:
            if dp[v] != -float('inf'):
                dp[u] = max(dp[u], dp[v] - (t_arr - t_dep))
                updated = True
        if not updated:
            break

    f_values_final = [-float('inf')] * n
    for i in range(1, n):
        max_arrival = -float('inf')
        for t_start in range(200):
            q = [(t_start, i)]
            while q:
                current_time, current_station = q.pop(0)
                if current_station == n:
                    max_arrival = max(max_arrival, current_time)
                    continue
                for t_dep, t_arr, from_node, to_node in trains:
                    if from_node == current_station and t_dep >= current_time:
                        q.append((t_arr, to_node))
        f_values_final[i-1] = max_arrival

    for i in range(1, n):
        max_arrival = -float('inf')
        for t_init in range(101):
            q = [(t_init, i)]
            while q:
                current_time, current_station = q.pop(0)
                if current_station == n:
                    max_arrival = max(max_arrival, current_time)
                    continue
                for t_dep, t_arr, from_node, to_node in trains:
                    if from_node == current_station and t_dep >= current_time:
                        q.append((t_arr, to_node))
        print(max_arrival if max_arrival > -float('inf') else "Unreachable")

solve()