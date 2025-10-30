def adventure(N, events):
    from collections import defaultdict

    potions = defaultdict(int)
    potion_actions = []
    max_potions = 0
    current_potions = 0

    for t, x in events:
        if t == 1:
            # Finding a potion
            potion_actions.append(1)  # Assume we pick it up
            potions[x] += 1
            current_potions += 1
            max_potions = max(max_potions, current_potions)
        elif t == 2:
            # Encountering a monster
            if potions[x] > 0:
                potions[x] -= 1
                current_potions -= 1
                potion_actions.append(0)  # No action needed for this event
            else:
                # Cannot defeat the monster
                return -1

    # Now we need to determine K_min
    K_min = max_potions
    result_actions = []
    current_potions = 0

    for t, x in events:
        if t == 1:
            if current_potions < K_min:
                result_actions.append(1)  # Pick it up
                potions[x] += 1
                current_potions += 1
            else:
                result_actions.append(0)  # Discard it
        elif t == 2:
            if potions[x] > 0:
                potions[x] -= 1
                current_potions -= 1
                result_actions.append(0)  # No action needed for this event

    return K_min, result_actions

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
events = [tuple(map(int, line.split())) for line in data[1:]]

result = adventure(N, events)

if result == -1:
    print(-1)
else:
    K_min, actions = result
    print(K_min)
    print(' '.join(map(str, actions)))