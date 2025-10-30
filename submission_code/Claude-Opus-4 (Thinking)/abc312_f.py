# YOUR CODE HERE
n, m = map(int, input().split())

pull_tabs = []
regular_cans = []
can_openers = []

for _ in range(n):
    t, x = map(int, input().split())
    if t == 0:
        pull_tabs.append(x)
    elif t == 1:
        regular_cans.append(x)
    else:  # t == 2
        can_openers.append(x)

# Sort in descending order for greedy selection
pull_tabs.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

# Precompute prefix sums for efficiency
def get_prefix_sums(arr):
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    return prefix

pull_tab_prefix = get_prefix_sums(pull_tabs)
regular_can_prefix = get_prefix_sums(regular_cans)
can_opener_prefix = get_prefix_sums(can_openers)

max_happiness = 0

# Try all possible distributions of M items
for num_pull_tabs in range(min(m + 1, len(pull_tabs) + 1)):
    for num_regular_cans in range(min(m - num_pull_tabs + 1, len(regular_cans) + 1)):
        num_can_openers = m - num_pull_tabs - num_regular_cans
        
        if num_can_openers < 0 or num_can_openers > len(can_openers):
            continue
        
        # Total capacity of selected can openers
        total_capacity = can_opener_prefix[num_can_openers]
        
        # We can open at most min(num_regular_cans, total_capacity) cans
        cans_opened = min(num_regular_cans, total_capacity)
        
        # Calculate total happiness
        happiness = pull_tab_prefix[num_pull_tabs] + regular_can_prefix[cans_opened]
        max_happiness = max(max_happiness, happiness)

print(max_happiness)