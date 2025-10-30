def solve():
    n, x, y = map(int, input().split())
    pt = []
    for _ in range(n - 1):
        pt.append(list(map(int, input().split())))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))

    for start_time in queries:
        current_time = start_time + x

        for i in range(n - 1):
            arrival_at_bus_stop = current_time
            p_i, t_i = pt[i]

            if arrival_at_bus_stop % p_i == 0:
                departure_from_bus_stop = arrival_at_bus_stop
            else:
                departure_from_bus_stop = arrival_at_bus_stop + (p_i - arrival_at_bus_stop % p_i)

            current_time = departure_from_bus_stop + t_i

        arrival_at_aoki = current_time + y
        print(arrival_at_aoki)

solve()