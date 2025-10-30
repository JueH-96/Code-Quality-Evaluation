N, M = map(int, input().split())
items = []
for i in range(N):
    T, X = map(int, input().split())
    items.append((T, X))

pull_tabs = [X for T, X in items if T == 0]
regular_cans = [X for T, X in items if T == 1]
can_openers = [X for T, X in items if T == 2]

pull_tabs.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

# Prefix sums for efficiency
pull_tab_sums = [0]
for x in pull_tabs:
    pull_tab_sums.append(pull_tab_sums[-1] + x)

regular_can_sums = [0]
for x in regular_cans:
    regular_can_sums.append(regular_can_sums[-1] + x)

can_opener_sums = [0]
for x in can_openers:
    can_opener_sums.append(can_opener_sums[-1] + x)

max_happiness = 0

for k2 in range(min(len(can_openers), M) + 1):
    max_openable = can_opener_sums[k2]
    
    for k1 in range(min(len(regular_cans), M - k2, max_openable) + 1):
        k0 = M - k1 - k2
        
        if k0 < 0 or k0 > len(pull_tabs):
            continue
            
        happiness = pull_tab_sums[k0] + regular_can_sums[k1]
        max_happiness = max(max_happiness, happiness)

print(max_happiness)