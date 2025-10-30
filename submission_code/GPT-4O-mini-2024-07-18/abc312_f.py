def max_happiness(N, M, items):
    pull_tabs = []
    regular_cans = []
    can_openers = []
    
    for T, X in items:
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular_cans.append(X)
        elif T == 2:
            can_openers.append(X)
    
    # Sort the lists in descending order to maximize happiness
    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    # Calculate the maximum happiness
    max_happiness = 0
    total_items = pull_tabs + regular_cans + can_openers
    total_items.sort(reverse=True)
    
    # Take the top M items from the sorted list
    max_happiness += sum(total_items[:M])
    
    # Now we need to consider the use of can openers
    # We will try to use can openers on regular cans
    can_openers_used = 0
    can_openers_count = 0
    
    for opener in can_openers:
        can_openers_count += opener
    
    # We can use can_openers_count can openers on regular cans
    # We need to take the minimum of available regular cans and can openers
    usable_regular_cans = min(len(regular_cans), can_openers_count)
    
    # Add the happiness from the regular cans that can be opened
    max_happiness += sum(regular_cans[:usable_regular_cans])
    
    return max_happiness

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
items = [tuple(map(int, line.split())) for line in data[1:N+1]]

result = max_happiness(N, M, items)
print(result)