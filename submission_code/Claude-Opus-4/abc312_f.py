# YOUR CODE HERE
N, M = map(int, input().split())

pull_tabs = []  # (happiness,)
regular_cans = []  # (happiness,)
can_openers = []  # (capacity,)

for _ in range(N):
    T, X = map(int, input().split())
    if T == 0:
        pull_tabs.append(X)
    elif T == 1:
        regular_cans.append(X)
    else:  # T == 2
        can_openers.append(X)

# Sort in descending order of value
pull_tabs.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

# Precompute prefix sums for pull_tabs and regular_cans
pull_tabs_prefix = [0]
for h in pull_tabs:
    pull_tabs_prefix.append(pull_tabs_prefix[-1] + h)

regular_cans_prefix = [0]
for h in regular_cans:
    regular_cans_prefix.append(regular_cans_prefix[-1] + h)

max_happiness = 0

# Try all possible numbers of can openers to take
for num_openers in range(min(M, len(can_openers)) + 1):
    # Calculate total capacity of selected can openers
    total_capacity = sum(can_openers[:num_openers])
    
    # Remaining items to select
    remaining = M - num_openers
    
    # We can open at most min(total_capacity, number of regular cans we take) cans
    # Try all possible distributions between pull-tabs and regular cans
    for num_regular in range(min(remaining, len(regular_cans)) + 1):
        num_pull_tabs = remaining - num_regular
        
        if num_pull_tabs > len(pull_tabs):
            continue
        
        # Calculate happiness from regular cans (limited by opener capacity)
        opened_cans = min(num_regular, total_capacity)
        regular_happiness = regular_cans_prefix[opened_cans]
        
        # Calculate happiness from pull-tabs
        pull_tab_happiness = pull_tabs_prefix[num_pull_tabs]
        
        total_happiness = regular_happiness + pull_tab_happiness
        max_happiness = max(max_happiness, total_happiness)

print(max_happiness)