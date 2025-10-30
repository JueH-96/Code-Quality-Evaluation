from collections import defaultdict

n = int(input())
events = []
for i in range(n):
    t, x = map(int, input().split())
    events.append((t, x))

# First check if it's possible at all
# Count potions available before each monster
potion_counts = defaultdict(int)
possible = True

for t, x in events:
    if t == 1:
        potion_counts[x] += 1
    else:  # t == 2
        if potion_counts[x] > 0:
            potion_counts[x] -= 1
        else:
            possible = False
            break

if not possible:
    print(-1)
else:
    # Find minimum K using greedy approach
    # Work backwards to determine latest pickup times
    
    # Find all monster positions for each type
    monster_positions = defaultdict(list)
    potion_positions = defaultdict(list)
    
    for i, (t, x) in enumerate(events):
        if t == 1:
            potion_positions[x].append(i)
        else:
            monster_positions[x].append(i)
    
    # For each monster type, assign potions to monsters greedily
    pickup_decisions = [0] * n  # 0 = don't pick up, 1 = pick up
    
    for monster_type in monster_positions:
        monsters = monster_positions[monster_type]
        potions = potion_positions[monster_type]
        
        # Assign potions to monsters (latest potion to latest monster)
        monster_idx = len(monsters) - 1
        potion_idx = len(potions) - 1
        
        while monster_idx >= 0 and potion_idx >= 0:
            monster_pos = monsters[monster_idx]
            
            # Find latest potion before this monster
            while potion_idx >= 0 and potions[potion_idx] >= monster_pos:
                potion_idx -= 1
            
            if potion_idx >= 0:
                pickup_decisions[potions[potion_idx]] = 1
                potion_idx -= 1
            
            monster_idx -= 1
    
    # Simulate the adventure to find K
    inventory = defaultdict(int)
    max_inventory = 0
    
    for i, (t, x) in enumerate(events):
        if t == 1:
            if pickup_decisions[i] == 1:
                inventory[x] += 1
                total_potions = sum(inventory.values())
                max_inventory = max(max_inventory, total_potions)
        else:  # t == 2
            inventory[x] -= 1
    
    print(max_inventory)
    
    # Output pickup decisions for potion events only
    result = []
    for i, (t, x) in enumerate(events):
        if t == 1:
            result.append(str(pickup_decisions[i]))
    
    print(' '.join(result))