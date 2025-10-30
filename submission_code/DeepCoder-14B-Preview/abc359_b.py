n = int(input())
a = list(map(int, input().split()))

color_indices = {}
for idx, val in enumerate(a):
    if val not in color_indices:
        color_indices[val] = []
    color_indices[val].append(idx)

count = 0
for color in color_indices:
    indices = color_indices[color]
    if abs(indices[1] - indices[0]) == 2:
        count += 1

print(count)