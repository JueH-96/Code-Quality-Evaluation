import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
events = [(int(data[i]), int(data[i+1])) for i in range(1, 2*N, 2)]

potion_count = defaultdict(int)
potion_actions = []
max_potions = 0
current_potions = 0

for t, x in events:
    if t == 1:
        potion_actions.append(0)  # Initially assume not picking up the potion
    elif t == 2:
        if potion_count[x] > 0:
            potion_count[x] -= 1
            current_potions -= 1
        else:
            print(-1)
            sys.exit()

max_potions = current_potions

# Determine the minimum number of potions needed
for i, (t, x) in enumerate(events):
    if t == 1:
        potion_count[x] += 1
        current_potions += 1
        potion_actions[i//2] = 1  # Change to picking up the potion
        max_potions = max(max_potions, current_potions)

# Check if we can defeat all monsters with the minimum number of potions
potion_count = defaultdict(int)
current_potions = 0

for t, x in events:
    if t == 1:
        if potion_actions.pop(0) == 1:
            potion_count[x] += 1
            current_potions += 1
    elif t == 2:
        if potion_count[x] > 0:
            potion_count[x] -= 1
            current_potions -= 1
        else:
            print(-1)
            sys.exit()

print(max_potions)
print(" ".join(map(str, potion_actions)))