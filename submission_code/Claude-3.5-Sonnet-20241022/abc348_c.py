N = int(input())
beans = []
for _ in range(N):
    a, c = map(int, input().split())
    beans.append((a, c))

# Group beans by color
color_groups = {}
for a, c in beans:
    if c not in color_groups:
        color_groups[c] = []
    color_groups[c].append(a)

# For each color, find minimum deliciousness
max_min_deliciousness = 0
for color in color_groups:
    min_deliciousness = min(color_groups[color])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

print(max_min_deliciousness)