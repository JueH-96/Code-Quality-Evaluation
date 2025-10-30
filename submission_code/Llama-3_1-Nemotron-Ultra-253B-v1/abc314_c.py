n, m = map(int, input().split())
s = list(input().strip())
c = list(map(int, input().split()))

color_groups = [[] for _ in range(m + 1)]
for idx in range(n):
    color = c[idx]
    color_groups[color].append(idx)

for color in range(1, m + 1):
    indices = color_groups[color]
    k = len(indices)
    if k <= 1:
        continue
    current = [s[i] for i in indices]
    rotated = [current[-1]] + current[:-1]
    for i, char in zip(indices, rotated):
        s[i] = char

print(''.join(s))