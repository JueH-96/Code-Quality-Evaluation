def solve():
    h, w, n = map(int, input().split())
    bars_data = []
    for _ in range(n):
        r, c, l = map(int, input().split())
        bars_data.append({'r': r, 'c': c, 'l': l})

    while True:
        moved_in_timestep = False
        initial_positions = [bar['r'] for bar in bars_data]
        next_bar_positions = list(initial_positions)

        occupied_at_start = set()
        for i in range(n):
            r, c, l = bars_data[i]['r'], bars_data[i]['c'], bars_data[i]['l']
            for col in range(c, c + l):
                occupied_at_start.add((r, col))

        new_positions_this_step = list(initial_positions)

        for i in range(n):
            r, c, l = bars_data[i]['r'], bars_data[i]['c'], bars_data[i]['l']
            if r < h:
                can_move_down = True
                for col in range(c, c + l):
                    if (r + 1, col) in occupied_at_start:
                        can_move_down = False
                        break
                if can_move_down:
                    new_positions_this_step[i] = r + 1

        if new_positions_this_step != initial_positions:
            moved_in_timestep = True
            for i in range(n):
                bars_data[i]['r'] = new_positions_this_step[i]

        if not moved_in_timestep:
            break

    for bar in bars_data:
        print(bar['r'])

solve()