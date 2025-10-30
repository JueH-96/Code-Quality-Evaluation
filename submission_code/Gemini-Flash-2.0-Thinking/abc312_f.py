from itertools import combinations

def solve():
    n, m = map(int, input().split())
    items_data = []
    for _ in range(n):
        t, x = map(int, input().split())
        items_data.append((t, x))

    max_happiness = 0
    for combo_indices in combinations(range(n), m):
        combination = [items_data[i] for i in combo_indices]

        pull_tabs = []
        regular_cans = []
        can_openers = []
        for t, x in combination:
            if t == 0:
                pull_tabs.append(x)
            elif t == 1:
                regular_cans.append(x)
            elif t == 2:
                can_openers.append(x)

        current_happiness = sum(pull_tabs)

        regular_cans.sort(reverse=True)
        can_openers.sort(reverse=True)

        opener_capacities = list(can_openers)
        opened_cans = 0
        for can_happiness in regular_cans:
            for i in range(len(opener_capacities)):
                if opener_capacities[i] > 0:
                    current_happiness += can_happiness
                    opener_capacities[i] -= 1
                    opened_cans += 1
                    break

        max_happiness = max(max_happiness, current_happiness)

    print(max_happiness)

solve()