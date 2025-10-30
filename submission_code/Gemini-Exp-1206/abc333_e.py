def solve():
    n = int(input())
    events = []
    for _ in range(n):
        events.append(list(map(int, input().split())))

    def check(actions):
        potions = {}
        max_potions = 0
        current_potions = 0
        action_idx = 0
        
        for t, x in events:
            if t == 1:
                if actions[action_idx] == 1:
                    potions[x] = potions.get(x, 0) + 1
                    current_potions += 1
                max_potions = max(max_potions, current_potions)
                action_idx += 1
            else:
                if x not in potions or potions[x] == 0:
                    return False, -1
                potions[x] -= 1
                current_potions -= 1
        return True, max_potions

    def find_min_k():
        
        potion_events_indices = [i for i, (t, _) in enumerate(events) if t == 1]
        num_potion_events = len(potion_events_indices)
        
        min_k = float('inf')
        best_actions = []

        for i in range(1 << num_potion_events):
            actions = []
            for j in range(num_potion_events):
                if (i >> j) & 1:
                    actions.append(1)
                else:
                    actions.append(0)
            
            possible, k = check(actions)
            if possible:
                if k < min_k:
                    min_k = k
                    best_actions = actions
                elif k == min_k:
                    
                    if not best_actions:
                        best_actions = actions
                    else:
                        
                        pass

        if min_k == float('inf'):
            print("-1")
        else:
            print(min_k)
            print(*best_actions)

    find_min_k()

solve()