# YOUR CODE HERE
N, M = map(int, input().split())
items = []
for i in range(N):
    T, X = map(int, input().split())
    items.append((T, X))

pull_tabs = []
regular_cans = []
can_openers = []
for T, X in items:
    if T == 0:
        pull_tabs.append(X)
    elif T == 1:
        regular_cans.append(X)
    else:
        can_openers.append(X)

pull_tabs.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

max_happiness = 0
for num_pull_tabs in range(min(M + 1, len(pull_tabs) + 1)):
    remaining_items = M - num_pull_tabs
    if remaining_items < 0:
        continue
    current_happiness = sum(pull_tabs[:num_pull_tabs])
    if remaining_items == 0:
        max_happiness = max(max_happiness, current_happiness)
        continue
    for num_can_openers in range(min(remaining_items + 1, len(can_openers) + 1)):
        num_regular_cans = remaining_items - num_can_openers
        if num_regular_cans < 0:
            continue
        if num_regular_cans > 0 and num_can_openers == 0:
            continue
        can_opener_capacity = sum(can_openers[:num_can_openers])
        if num_regular_cans > can_opener_capacity:
            continue
        current_happiness = sum(pull_tabs[:num_pull_tabs]) + sum(regular_cans[:num_regular_cans])
        max_happiness = max(max_happiness, current_happiness)

print(max_happiness)