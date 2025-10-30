n = int(input())
events = []
for _ in range(n):
    t, x = map(int, input().split())
    events.append((t, x))

def can_solve_with_k(k):
    # Simulate the process with at most k potions
    inventory = {}  # type -> count
    total_potions = 0
    actions = []
    
    # Calculate future monster requirements
    future_needs = {}  # type -> count of future monsters of this type
    for i in range(n - 1, -1, -1):
        t, x = events[i]
        if t == 2:
            future_needs[x] = future_needs.get(x, 0) + 1
    
    for i, (t, x) in enumerate(events):
        if t == 1:
            # Decide whether to pick up this potion
            current_count = inventory.get(x, 0)
            future_count = future_needs.get(x, 0)
            
            if future_count > current_count and total_potions < k:
                # Pick up the potion
                inventory[x] = inventory.get(x, 0) + 1
                total_potions += 1
                actions.append(1)
            else:
                actions.append(0)
        else:  # t == 2
            if inventory.get(x, 0) == 0:
                return False, []
            inventory[x] -= 1
            total_potions -= 1
            future_needs[x] -= 1
    
    return True, actions

# First check if it's possible at all
possible, _ = can_solve_with_k(n)
if not possible:
    print(-1)
else:
    # Binary search on k
    left, right = 1, n
    best_k = -1
    best_actions = []
    
    while left <= right:
        mid = (left + right) // 2
        possible, actions = can_solve_with_k(mid)
        if possible:
            best_k = mid
            best_actions = actions
            right = mid - 1
        else:
            left = mid + 1
    
    print(best_k)
    print(' '.join(map(str, best_actions)))