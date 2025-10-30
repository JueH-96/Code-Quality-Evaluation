from collections import defaultdict

# Read input
N = int(input())
beans = []
for _ in range(N):
    A, C = map(int, input().split())
    beans.append((A, C))

# Group beans by color
color_to_deliciousness = defaultdict(list)
for deliciousness, color in beans:
    color_to_deliciousness[color].append(deliciousness)

# Sort each color's deliciousness in ascending order
for color in color_to_deliciousness:
    color_to_deliciousness[color].sort()

# Find the maximum of the minimum deliciousness
max_min_deliciousness = 0
for color, deliciousness_list in color_to_deliciousness.items():
    max_min_deliciousness = max(max_min_deliciousness, deliciousness_list[0])

print(max_min_deliciousness)