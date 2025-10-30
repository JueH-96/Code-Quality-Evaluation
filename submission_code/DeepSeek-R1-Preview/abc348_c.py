n = int(input())
color_min = {}

for _ in range(n):
    a, c = map(int, input().split())
    if c in color_min:
        if a < color_min[c]:
            color_min[c] = a
    else:
        color_min[c] = a

print(max(color_min.values()))