from collections import defaultdict
from heapq import heappush, heappop

N = int(input())
events = [tuple(map(int, input().split())) for _ in range(N)]

# Sort events by their type and index
events.sort(key=lambda x: (x[0], x[1], -x[0], -x[1]))

# Initialize data structures
potion_counts = defaultdict(int)
max_potions = 0
actions = []
potion_heap = []
monster_queue = []

# Process events
for t, x in events:
    if t == 1:  # Found a potion
        if potion_counts[x] < max_potions:
            heappush(potion_heap, x)
            actions.append(1)
        else:
            actions.append(0)
        potion_counts[x] += 1
        max_potions = max(max_potions, potion_counts[x])
    elif t == 2:  # Encounter a monster
        if potion_counts[x] == 0:
            if not potion_heap:
                print(-1)
                return
            x = heappop(potion_heap)
            potion_counts[x] -= 1
        else:
            potion_counts[x] -= 1

# Print results
print(max_potions)
print(' '.join(map(str, actions)))