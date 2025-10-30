n = int(input())
events = []
for _ in range(n):
    t, x = map(int, input().split())
    events.append((t, x))

# Check if it's possible at all
def is_possible():
    potions = {}
    for t, x in events:
        if t == 1:
            potions[x] = potions.get(x, 0) + 1
        else:
            if potions.get(x, 0) == 0:
                return False
            potions[x] -= 1
    return True

if not is_possible():
    print(-1)
else:
    # Binary search for minimum K
    def can_complete_with_k(k):
        # Count remaining monsters of each type after each position
        remaining_monsters = [{} for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            remaining_monsters[i] = remaining_monsters[i + 1].copy()
            if events[i][0] == 2:
                x = events[i][1]
                remaining_monsters[i][x] = remaining_monsters[i].get(x, 0) + 1
        
        inventory = {}
        actions = []
        
        for i, (t, x) in enumerate(events):
            if t == 1:
                # Found a potion
                current_count = sum(inventory.values())
                monsters_need = remaining_monsters[i + 1].get(x, 0)
                current_have = inventory.get(x, 0)
                
                # Pick up if we have space and need more of this type
                if current_count < k and current_have < monsters_need:
                    inventory[x] = inventory.get(x, 0) + 1
                    actions.append(1)
                else:
                    actions.append(0)
            else:
                # Monster
                if inventory.get(x, 0) == 0:
                    return False, []
                inventory[x] -= 1
        
        return True, actions
    
    left, right = 1, n
    result_k = n
    result_actions = []
    
    while left <= right:
        mid = (left + right) // 2
        success, actions = can_complete_with_k(mid)
        if success:
            result_k = mid
            result_actions = actions
            right = mid - 1
        else:
            left = mid + 1
    
    print(result_k)
    print(' '.join(map(str, result_actions)))