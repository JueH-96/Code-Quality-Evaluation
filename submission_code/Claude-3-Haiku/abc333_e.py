import sys

N = int(input())
events = []
for _ in range(N):
    t, x = map(int, input().split())
    events.append((t, x))

# Sort events by type, then by monster type
events.sort()

# Initialize variables
potions = [0] * (N + 1)
defeated_monsters = [False] * (N + 1)
max_potions = 0
actions = [0] * N

# Simulate the adventure
for i, (t, x) in enumerate(events):
    if t == 1:  # Find a potion
        potions[x] += 1
        actions[i] = 1
        max_potions = max(max_potions, sum(potions))
    else:  # Encounter a monster
        if potions[x] > 0:
            potions[x] -= 1
            defeated_monsters[x] = True
        else:
            print(-1)
            return

# Check if all monsters were defeated
if all(defeated_monsters):
    print(max_potions)
    print(" ".join(map(str, actions)))
else:
    print(-1)