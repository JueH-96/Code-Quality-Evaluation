n = int(input())
a = list(map(int, input().split()))
color_indices = {}

for idx, color in enumerate(a):
    if color not in color_indices:
        color_indices[color] = []
    color_indices[color].append(idx)

count = 0
for i in range(1, n + 1):
    indices = color_indices[i]
    if indices[1] - indices[0] == 2:
        count += 1

print(count)