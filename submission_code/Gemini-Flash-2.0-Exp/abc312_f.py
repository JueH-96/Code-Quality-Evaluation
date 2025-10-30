def solve():
    n, m = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    max_happiness = 0
    for i in range(1 << n):
        if bin(i).count('1') != m:
            continue

        current_happiness = 0
        selected_items = []
        for j in range(n):
            if (i >> j) & 1:
                selected_items.append(items[j])

        pull_tab_happiness = 0
        regular_cans = []
        can_openers = []

        for item_type, value in selected_items:
            if item_type == 0:
                pull_tab_happiness += value
            elif item_type == 1:
                regular_cans.append(value)
            else:
                can_openers.append(value)

        current_happiness += pull_tab_happiness
        regular_cans.sort(reverse=True)
        can_openers.sort(reverse=True)

        num_cans_opened = 0
        opener_index = 0
        can_index = 0

        while opener_index < len(can_openers) and can_index < len(regular_cans):
            num_openings = min(can_openers[opener_index], len(regular_cans) - can_index)
            for k in range(num_openings):
                current_happiness += regular_cans[can_index + k]
            can_index += num_openings
            opener_index += 1

        max_happiness = max(max_happiness, current_happiness)

    print(max_happiness)

solve()