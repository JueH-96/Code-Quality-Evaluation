# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
beans = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

color_to_deliciousness = defaultdict(list)
for deliciousness, color in beans:
    color_to_deliciousness[color].append(deliciousness)

max_min_deliciousness = 0
for color in color_to_deliciousness:
    min_deliciousness = min(color_to_deliciousness[color])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

print(max_min_deliciousness)