def solve():
    n = int(input())
    events = []
    for _ in range(n):
        events.append(list(map(int, input().split())))

    def check(actions):
        potions = {}
        max_potions = 0
        action_idx = 0
        
        for t, x in events:
            if t == 1:
                if actions[action_idx] == 1:
                    potions[x] = potions.get(x, 0) + 1
                action_idx += 1
                max_potions = max(max_potions, sum(potions.values()))
            elif t == 2:
                if potions.get(x, 0) > 0:
                    potions[x] -= 1
                else:
                    return False, -1
        return True, max_potions

    potion_indices = [i for i, (t, _) in enumerate(events) if t == 1]
    num_potions = len(potion_indices)
    
    min_k = float('inf')
    best_actions = None

    for i in range(2**num_potions):
        actions = []
        for j in range(num_potions):
            if (i >> j) & 1:
                actions.append(1)
            else:
                actions.append(0)
        
        success, max_potions = check(actions)
        
        if success:
            if max_potions < min_k:
                min_k = max_potions
                best_actions = actions

    if best_actions is None:
        print("-1")
    else:
        print(min_k)
        print(*best_actions)

solve()