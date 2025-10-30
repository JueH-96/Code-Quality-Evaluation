import sys
import bisect

input = sys.stdin.read
data = input().split()

W = int(data[0])
H = int(data[1])
N = int(data[2])

strawberries = []
index = 3
for _ in range(N):
    p = int(data[index])
    q = int(data[index + 1])
    strawberries.append((p, q))
    index += 2

A = int(data[index])
a_cuts = list(map(int, data[index + 1:index + 1 + A]))
index += 1 + A

B = int(data[index])
b_cuts = list(map(int, data[index + 1:index + 1 + B]))
index += 1 + B

# Add boundaries for the cake
x_cuts = [0] + a_cuts + [W]
y_cuts = [0] + b_cuts + [H]

# Sort the cuts (although they should already be sorted by constraints)
x_cuts.sort()
y_cuts.sort()

# Create a 2D grid to count strawberries in each piece
strawberry_count = [[0] * (B + 1) for _ in range(A + 1)]

# Place each strawberry in the correct piece
for p, q in strawberries:
    x_idx = bisect.bisect_right(x_cuts, p) - 1
    y_idx = bisect.bisect_right(y_cuts, q) - 1
    strawberry_count[x_idx][y_idx] += 1

# Find the minimum and maximum number of strawberries in any piece
min_strawberries = float('inf')
max_strawberries = float('-inf')

for row in strawberry_count:
    for count in row:
        if count < min_strawberries:
            min_strawberries = count
        if count > max_strawberries:
            max_strawberries = count

print(min_strawberries, max_strawberries)