# YOUR CODE HERE
import sys
from bisect import bisect_left, bisect_right

# Read input
W, H = map(int, sys.stdin.readline().split())
N = int(sys.stdin.readline())
strawberries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
A = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
B = int(sys.stdin.readline())
b = list(map(int, sys.stdin.readline().split()))

# Sort the strawberries by x and y coordinates
strawberries.sort()

# Count strawberries in each vertical strip
vertical_strips = [0] * (A + 1)
current_strip = 0
for x, y in strawberries:
    while current_strip < A and x > a[current_strip]:
        current_strip += 1
    vertical_strips[current_strip] += 1

# Count strawberries in each horizontal strip
horizontal_strips = [0] * (B + 1)
current_strip = 0
for x, y in strawberries:
    while current_strip < B and y > b[current_strip]:
        current_strip += 1
    horizontal_strips[current_strip] += 1

# Calculate the minimum and maximum number of strawberries in any piece
min_strawberries = float('inf')
max_strawberries = 0

for i in range(A + 1):
    for j in range(B + 1):
        count = vertical_strips[i] + horizontal_strips[j]
        min_strawberries = min(min_strawberries, count)
        max_strawberries = max(max_strawberries, count)

print(min_strawberries, max_strawberries)