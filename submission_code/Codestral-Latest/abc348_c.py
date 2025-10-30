import sys
from collections import defaultdict

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
beans = []

for i in range(1, N * 2 + 1, 2):
    A_i = int(data[i])
    C_i = int(data[i + 1])
    beans.append((A_i, C_i))

# Group beans by color
color_dict = defaultdict(list)
for A_i, C_i in beans:
    color_dict[C_i].append(A_i)

# Find the maximum minimum deliciousness
max_min_deliciousness = 0
for color in color_dict:
    min_deliciousness = min(color_dict[color])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

# Print the result
print(max_min_deliciousness)