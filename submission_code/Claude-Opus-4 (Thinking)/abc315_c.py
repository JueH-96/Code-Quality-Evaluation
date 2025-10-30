# YOUR CODE HERE
n = int(input())
cups = []
for _ in range(n):
    f, s = map(int, input().split())
    cups.append((f, s))

# Find maximum satisfaction
max_satisfaction = 0

# Group by flavor
flavor_groups = {}
for f, s in cups:
    if f not in flavor_groups:
        flavor_groups[f] = []
    flavor_groups[f].append(s)

# Sort deliciousness values in descending order for each flavor
for f in flavor_groups:
    flavor_groups[f].sort(reverse=True)

# Case 1: Same flavor
for f, deliciousness_list in flavor_groups.items():
    if len(deliciousness_list) >= 2:
        s = deliciousness_list[0]
        t = deliciousness_list[1]
        satisfaction = s + t // 2
        max_satisfaction = max(max_satisfaction, satisfaction)

# Case 2: Different flavors
# Get the top 2 deliciousness values for each flavor
candidates = []
for f, deliciousness_list in flavor_groups.items():
    for i in range(min(2, len(deliciousness_list))):
        candidates.append((deliciousness_list[i], f))

# Sort by deliciousness in descending order
candidates.sort(reverse=True)

# Find the best pair with different flavors
for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
        s1, f1 = candidates[i]
        s2, f2 = candidates[j]
        if f1 != f2:
            satisfaction = s1 + s2
            max_satisfaction = max(max_satisfaction, satisfaction)
            break  # We found the best pair for this i

print(max_satisfaction)