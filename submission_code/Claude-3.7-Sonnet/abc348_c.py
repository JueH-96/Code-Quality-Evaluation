n = int(input())
min_deliciousness_per_color = {}

for _ in range(n):
    deliciousness, color = map(int, input().split())
    if color not in min_deliciousness_per_color:
        min_deliciousness_per_color[color] = deliciousness
    else:
        min_deliciousness_per_color[color] = min(min_deliciousness_per_color[color], deliciousness)

print(max(min_deliciousness_per_color.values()))