n = int(input())
cups = []
for i in range(n):
    f, s = map(int, input().split())
    cups.append((f, s))

max_satisfaction = 0

# Sort cups by deliciousness in descending order
sorted_cups = sorted(cups, key=lambda x: x[1], reverse=True)

# Different flavors: find the first two cups with different flavors
first_flavor = sorted_cups[0][0]
first_deliciousness = sorted_cups[0][1]
for i in range(1, n):
    if sorted_cups[i][0] != first_flavor:
        second_deliciousness = sorted_cups[i][1]
        max_satisfaction = max(max_satisfaction, first_deliciousness + second_deliciousness)
        break

# Same flavors
from collections import defaultdict
flavor_to_deliciousness = defaultdict(list)
for f, s in cups:
    flavor_to_deliciousness[f].append(s)

for flavor, deliciousness_list in flavor_to_deliciousness.items():
    if len(deliciousness_list) >= 2:
        deliciousness_list.sort(reverse=True)
        s = deliciousness_list[0]
        t = deliciousness_list[1]
        max_satisfaction = max(max_satisfaction, s + t // 2)

print(max_satisfaction)