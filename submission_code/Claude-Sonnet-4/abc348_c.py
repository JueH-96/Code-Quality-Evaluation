# YOUR CODE HERE
n = int(input())
color_to_deliciousness = {}

for _ in range(n):
    a, c = map(int, input().split())
    if c not in color_to_deliciousness:
        color_to_deliciousness[c] = []
    color_to_deliciousness[c].append(a)

max_min_deliciousness = 0
for color, deliciousness_list in color_to_deliciousness.items():
    min_deliciousness = min(deliciousness_list)
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

print(max_min_deliciousness)