# YOUR CODE HERE
n = int(input())
cups = []
for _ in range(n):
    f, s = map(int, input().split())
    cups.append((f, s))

# Group cups by flavor
flavor_groups = {}
for f, s in cups:
    if f not in flavor_groups:
        flavor_groups[f] = []
    flavor_groups[f].append(s)

# Sort each flavor group in descending order
for f in flavor_groups:
    flavor_groups[f].sort(reverse=True)

max_satisfaction = 0

# Case 1: Same flavor
for f in flavor_groups:
    if len(flavor_groups[f]) >= 2:
        s = flavor_groups[f][0]
        t = flavor_groups[f][1]
        satisfaction = s + t // 2
        max_satisfaction = max(max_satisfaction, satisfaction)

# Case 2: Different flavors
# Get the best cup from each flavor
best_from_each_flavor = []
for f in flavor_groups:
    best_from_each_flavor.append((flavor_groups[f][0], f))

# Sort by deliciousness in descending order
best_from_each_flavor.sort(reverse=True)

# Try all pairs of different flavors
for i in range(len(best_from_each_flavor)):
    for j in range(i + 1, len(best_from_each_flavor)):
        s = best_from_each_flavor[i][0]
        t = best_from_each_flavor[j][0]
        satisfaction = s + t
        max_satisfaction = max(max_satisfaction, satisfaction)

print(max_satisfaction)