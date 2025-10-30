import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
events = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

# Initialize variables
potion_count = [0] * (N + 1)
max_potions = 0
current_potions = 0
actions = [0] * N

# Process events
for i, (t, x) in enumerate(events):
    if t == 1:
        # Find a potion
        potion_count[x] += 1
        current_potions += 1
        actions[i] = 1
        max_potions = max(max_potions, current_potions)
    elif t == 2:
        # Encounter a monster
        if potion_count[x] > 0:
            # Use a potion to defeat the monster
            potion_count[x] -= 1
            current_potions -= 1
        else:
            # Cannot defeat the monster
            print(-1)
            sys.exit()

# Print the result
print(max_potions)
for action in actions:
    print(action, end=' ')
print()