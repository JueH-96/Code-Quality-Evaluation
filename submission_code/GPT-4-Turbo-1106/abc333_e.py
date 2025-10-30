from collections import defaultdict

def solve_adventure(N, events):
    potion_counts = defaultdict(int)
    max_potions = 0
    actions = []

    # First pass to check if it's possible to defeat all monsters
    for t, x in events:
        if t == 1:
            potion_counts[x] += 1
            max_potions = max(max_potions, potion_counts[x])
        elif t == 2:
            if potion_counts[x] == 0:
                return -1, []
            potion_counts[x] -= 1

    # Second pass to determine the minimum K and actions
    potion_counts.clear()
    min_K = max_potions
    for t, x in events:
        if t == 1:
            if potion_counts[x] < min_K:
                potion_counts[x] += 1
                actions.append(1)
            else:
                actions.append(0)
        elif t == 2:
            potion_counts[x] -= 1

    return min_K, actions

# Read input
N = int(input().strip())
events = [tuple(map(int, input().split())) for _ in range(N)]

# Solve the problem
min_K, actions = solve_adventure(N, events)

# Write output
if min_K == -1:
    print(min_K)
else:
    print(min_K)
    print(" ".join(map(str, actions)))