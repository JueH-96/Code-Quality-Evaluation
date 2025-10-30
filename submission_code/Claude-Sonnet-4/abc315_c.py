n = int(input())
cups = []
for i in range(n):
    f, s = map(int, input().split())
    cups.append((f, s))

# Sort all cups by deliciousness in descending order
all_sorted = sorted(cups, key=lambda x: x[1], reverse=True)

max_satisfaction = 0

# For different flavors: find the best pair with different flavors
for i in range(n):
    for j in range(i + 1, n):
        if all_sorted[i][0] != all_sorted[j][0]:
            # Found first pair with different flavors
            max_satisfaction = max(max_satisfaction, all_sorted[i][1] + all_sorted[j][1])
            break
    if max_satisfaction > 0:
        break

# For same flavor: find best pair within each flavor
flavor_groups = {}
for f, s in cups:
    if f not in flavor_groups:
        flavor_groups[f] = []
    flavor_groups[f].append(s)

for flavor, deliciousness_list in flavor_groups.items():
    if len(deliciousness_list) >= 2:
        deliciousness_list.sort(reverse=True)
        s = deliciousness_list[0]
        t = deliciousness_list[1]
        satisfaction = s + t // 2
        max_satisfaction = max(max_satisfaction, satisfaction)

print(max_satisfaction)