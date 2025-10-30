# YOUR CODE HERE
N = int(input())
ice_creams = [list(map(int, input().split())) for _ in range(N)]

# Group ice creams by flavor
flavors = {}
for i, (f, s) in enumerate(ice_creams):
    if f not in flavors:
        flavors[f] = []
    flavors[f].append((s, i))

# Sort ice creams within each flavor group by deliciousness
for f in flavors:
    flavors[f].sort(reverse=True)

# Find the two most delicious ice creams of different flavors
max_diff = 0
max_s1, max_s2 = 0, 0
for f in flavors:
    if flavors[f][0][0] > max_s1:
        max_s2 = max_s1
        max_s1 = flavors[f][0][0]
    elif flavors[f][0][0] > max_s2:
        max_s2 = flavors[f][0][0]
max_diff = max_s1 + max_s2

# Find the maximum satisfaction for same flavor
max_same = 0
for f in flavors:
    if len(flavors[f]) >= 2:
        s1, s2 = flavors[f][0][0], flavors[f][1][0]
        max_same = max(max_same, s1 + s2 // 2)

# Output the maximum of the two cases
print(max(max_diff, max_same))