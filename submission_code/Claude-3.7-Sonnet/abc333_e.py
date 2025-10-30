def main():
    n = int(input())
    events = []
    for i in range(n):
        t, x = map(int, input().split())
        events.append((t, x))
    
    result = solve(events)
    if result == -1:
        print(-1)
    else:
        max_potions, pickup_actions = result
        print(max_potions)
        print(' '.join(map(str, pickup_actions)))

def solve(events):
    # First check if it's possible to defeat all monsters
    # by simulating picking up all potions
    potions_held = {}  # Maps potion type to the number of potions held
    
    for t, x in events:
        if t == 1:  # Found a potion
            potions_held[x] = potions_held.get(x, 0) + 1
        else:  # Encountered a monster
            if potions_held.get(x, 0) <= 0:
                return -1  # Impossible to defeat this monster
            potions_held[x] -= 1
    
    # If we're here, it's possible to defeat all monsters
    
    # Count the number of monsters of each type that will be encountered in the future
    monsters_remaining = {}  # Maps monster type to the number of monsters remaining
    
    for i in range(len(events) - 1, -1, -1):
        t, x = events[i]
        if t == 2:  # Monster
            monsters_remaining[x] = monsters_remaining.get(x, 0) + 1
    
    # Simulate the adventure optimally
    potions_held = {}  # Maps potion type to the number of potions currently held
    actions = {}  # Maps event index to action (1 for pick up, 0 for discard)
    
    for i, (t, x) in enumerate(events):
        if t == 1:  # Found a potion
            if potions_held.get(x, 0) < monsters_remaining.get(x, 0):
                # We need this potion
                potions_held[x] = potions_held.get(x, 0) + 1
                actions[i] = 1  # Pick up
            else:
                actions[i] = 0  # Discard
        else:  # Encountered a monster
            potions_held[x] -= 1
            monsters_remaining[x] -= 1
    
    # Calculate K (maximum number of potions held)
    potions_held = {}
    max_potions = 0
    total_potions = 0
    
    for i, (t, x) in enumerate(events):
        if t == 1:  # Found a potion
            if actions[i] == 1:
                potions_held[x] = potions_held.get(x, 0) + 1
                total_potions += 1
            
            max_potions = max(max_potions, total_potions)
        
        else:  # Encountered a monster
            potions_held[x] -= 1
            total_potions -= 1
    
    # Construct the answer
    pickup_actions = []
    for i, (t, x) in enumerate(events):
        if t == 1:
            pickup_actions.append(actions[i])
    
    return max_potions, pickup_actions

if __name__ == "__main__":
    main()