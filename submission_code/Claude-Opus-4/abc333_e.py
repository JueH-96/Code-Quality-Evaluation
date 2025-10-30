def can_complete_with_k(events, k):
    n = len(events)
    potions = {}  # type -> count
    actions = []
    
    # First, check what monsters we'll encounter
    future_monsters = {}
    for i in range(n):
        t, x = events[i]
        if t == 2:
            future_monsters[x] = future_monsters.get(x, 0) + 1
    
    current_potions = 0
    for i in range(n):
        t, x = events[i]
        
        if t == 1:  # Found potion
            # Check if we need this potion type for future monsters
            need_count = future_monsters.get(x, 0) - potions.get(x, 0)
            
            if need_count > 0 and current_potions < k:
                # Pick it up
                potions[x] = potions.get(x, 0) + 1
                current_potions += 1
                actions.append(1)
            else:
                # Don't pick it up
                actions.append(0)
        else:  # t == 2, encounter monster
            if potions.get(x, 0) == 0:
                return False, []
            potions[x] -= 1
            current_potions -= 1
            future_monsters[x] -= 1
    
    return True, actions

def solve():
    n = int(input())
    events = []
    for _ in range(n):
        t, x = map(int, input().split())
        events.append((t, x))
    
    # First check if it's possible at all
    # Count potions and monsters of each type
    potion_count = {}
    monster_count = {}
    
    for t, x in events:
        if t == 1:
            potion_count[x] = potion_count.get(x, 0) + 1
        else:
            monster_count[x] = monster_count.get(x, 0) + 1
    
    # Check if we have enough potions of each type
    for monster_type, count in monster_count.items():
        if potion_count.get(monster_type, 0) < count:
            print(-1)
            return
    
    # Binary search for minimum K
    left, right = 1, n
    best_k = n
    best_actions = []
    
    while left <= right:
        mid = (left + right) // 2
        possible, actions = can_complete_with_k(events, mid)
        
        if possible:
            best_k = mid
            best_actions = actions
            right = mid - 1
        else:
            left = mid + 1
    
    # Need to verify with a more careful approach for the best_k
    # Use a greedy strategy that looks ahead
    potions = {}
    actions = []
    current_total = 0
    
    # Count remaining monsters after each position
    remaining_monsters = {}
    for i in range(n-1, -1, -1):
        t, x = events[i]
        if t == 2:
            remaining_monsters[x] = remaining_monsters.get(x, 0) + 1
    
    for i in range(n):
        t, x = events[i]
        
        if t == 1:  # Found potion
            # Check if we need this potion
            need = remaining_monsters.get(x, 0) > potions.get(x, 0)
            
            if need and current_total < best_k:
                potions[x] = potions.get(x, 0) + 1
                current_total += 1
                actions.append(1)
            else:
                actions.append(0)
        else:  # Encounter monster
            potions[x] -= 1
            current_total -= 1
            remaining_monsters[x] -= 1
    
    print(best_k)
    print(' '.join(map(str, actions)))

solve()