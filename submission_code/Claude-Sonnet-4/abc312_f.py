# YOUR CODE HERE
n, m = map(int, input().split())
items = []
for i in range(n):
    t, x = map(int, input().split())
    items.append((t, x))

# Separate items by type
pull_tabs = []  # type 0
regular_cans = []  # type 1
can_openers = []  # type 2

for t, x in items:
    if t == 0:
        pull_tabs.append(x)
    elif t == 1:
        regular_cans.append(x)
    else:
        can_openers.append(x)

# Sort in descending order for greedy selection
pull_tabs.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

max_happiness = 0

# Try different numbers of can openers
max_openers = min(m, len(can_openers))

for num_openers in range(max_openers + 1):
    if num_openers > m:
        continue
    
    # Calculate total capacity from selected can openers
    total_capacity = sum(can_openers[:num_openers])
    
    # Select regular cans up to capacity
    num_regular = min(total_capacity, len(regular_cans), m - num_openers)
    regular_happiness = sum(regular_cans[:num_regular])
    
    # Fill remaining slots with pull-tab cans
    remaining_slots = m - num_openers - num_regular
    num_pull_tabs = min(remaining_slots, len(pull_tabs))
    pull_tab_happiness = sum(pull_tabs[:num_pull_tabs])
    
    total_happiness = regular_happiness + pull_tab_happiness
    max_happiness = max(max_happiness, total_happiness)

print(max_happiness)