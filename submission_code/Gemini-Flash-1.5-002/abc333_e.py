# YOUR CODE HERE
N = int(input())
events = []
for i in range(N):
    t, x = map(int, input().split())
    events.append((t, x))

potions = {}
actions = {}
max_potions = 0
defeated = False

for i, (t, x) in enumerate(events):
    if t == 1:
        if x not in potions:
            potions[x] = 0
        actions[i] = 0
    else:
        if x not in potions or potions[x] == 0:
            defeated = True
            break
        else:
            potions[x] -= 1

if defeated:
    print(-1)
else:
    min_k = float('inf')
    best_actions = []
    
    q = [(0, {}, 0)] # (current_event_index, current_potions, max_potions_so_far)
    visited = set()

    while q:
        curr_idx, curr_potions, curr_max_potions = q.pop(0)
        
        if curr_idx == N:
            if curr_max_potions < min_k:
                min_k = curr_max_potions
                best_actions = []
                for i in actions:
                    if i in curr_potions and curr_potions[i] == 1:
                        best_actions.append(1)
                    else:
                        best_actions.append(0)
            continue

        t, x = events[curr_idx]
        if t == 1:
            new_potions = curr_potions.copy()
            if x not in new_potions:
                new_potions[x] = 0
            new_potions[curr_idx] = 1
            max_potions_so_far = max(curr_max_potions, sum(new_potions.values()))
            q.append((curr_idx + 1, new_potions, max_potions_so_far))

            new_potions = curr_potions.copy()
            new_potions[curr_idx] = 0
            max_potions_so_far = max(curr_max_potions, sum(new_potions.values()))
            q.append((curr_idx + 1, new_potions, max_potions_so_far))
        else:
            new_potions = curr_potions.copy()
            if x in new_potions and new_potions[x] > 0:
                new_potions[x] -= 1
                max_potions_so_far = max(curr_max_potions, sum(new_potions.values()))
                q.append((curr_idx + 1, new_potions, max_potions_so_far))

    print(min_k)
    
    pick_up_actions = []
    idx = 0
    for i in range(N):
        if events[i][0] == 1:
            pick_up_actions.append(best_actions[idx])
            idx += 1
    print(*pick_up_actions)