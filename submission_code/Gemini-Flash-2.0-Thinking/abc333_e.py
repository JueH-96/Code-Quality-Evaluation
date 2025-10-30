def solve():
    n = int(input())
    events = []
    for _ in range(n):
        events.append(list(map(int, input().split())))

    potion_indices = [i for i, (t, _) in enumerate(events) if t == 1]
    num_potions = len(potion_indices)

    def check_strategy(picks):
        potions = {}
        max_potions = 0
        current_potions = 0

        pick_actions = []

        potion_idx_counter = 0
        for i in range(n):
            t, x = events[i]
            if t == 1:
                if picks[potion_idx_counter]:
                    potions[x] = potions.get(x, 0) + 1
                    current_potions += 1
                pick_actions.append(1 if picks[potion_idx_counter] else 0)
                potion_idx_counter += 1
            elif t == 2:
                if potions.get(x, 0) > 0:
                    potions[x] -= 1
                    current_potions -= 1
                else:
                    return False, -1, []
            max_potions = max(max_potions, current_potions)
        return True, max_potions, pick_actions

    possible_strategies = []

    def find_strategies(index, current_picks):
        if index == num_potions:
            is_possible, max_k, actions = check_strategy(current_picks)
            if is_possible:
                possible_strategies.append((max_k, actions))
            return

        find_strategies(index + 1, current_picks + [True])
        find_strategies(index + 1, current_picks + [False])

    find_strategies(0, [])

    if not possible_strategies:
        print(-1)
        return

    min_k = min(k for k, _ in possible_strategies)

    for k, actions in possible_strategies:
        if k == min_k:
            print(min_k)
            print(*actions)
            return

solve()