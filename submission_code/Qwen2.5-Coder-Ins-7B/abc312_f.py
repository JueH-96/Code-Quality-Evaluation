import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

items = []
for i in range(N):
    T = int(data[2 + 2 * i])
    X = int(data[3 + 2 * i])
    items.append((T, X))

# Separate items into categories
pull_tab_cans = [X for T, X in items if T == 0]
regular_cans = [X for T, X in items if T == 1]
can_openers = [X for T, X in items if T == 2]

# Sort items by happiness in descending order
pull_tab_cans.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort(reverse=True)

# Calculate the maximum happiness
max_happiness = 0

# Use all can openers to open regular cans
for i in range(min(len(can_openers), M)):
    if i < len(regular_cans):
        max_happiness += regular_cans[i]

# Use remaining can openers to open pull-tab cans
for i in range(min(len(can_openers) - M, len(pull_tab_cans))):
    max_happiness += pull_tab_cans[i]

# Add the happiness of the remaining pull-tab cans
max_happiness += sum(pull_tab_cans[M + len(can_openers):])

# Add the happiness of the remaining regular cans
max_happiness += sum(regular_cans[M + len(can_openers):])

# Add the happiness of the remaining can openers
max_happiness += sum(can_openers[M + len(regular_cans):])

print(max_happiness)