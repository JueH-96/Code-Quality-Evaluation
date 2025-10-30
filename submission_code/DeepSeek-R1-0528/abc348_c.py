n = int(input().strip())
min_per_color = {}
for _ in range(n):
    a, c = map(int, input().split())
    if c not in min_per_color or a < min_per_color[c]:
        min_per_color[c] = a
print(max(min_per_color.values()))