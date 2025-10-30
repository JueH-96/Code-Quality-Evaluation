import bisect
from collections import defaultdict

# Read input
W, H = map(int, input().split())
N = int(input())
strawberries = [tuple(map(int, input().split())) for _ in range(N)]
A = int(input())
a = list(map(int, input().split()))
B = int(input())
b = list(map(int, input().split()))

# Prepare extended arrays for binary search
extended_a = [0] + a + [W]
extended_b = [0] + b + [H]

# Count strawberries in each cell
counts = defaultdict(int)
for p, q in strawberries:
    k = bisect.bisect_right(extended_a, p) - 1
    m = bisect.bisect_right(extended_b, q) - 1
    counts[(k, m)] += 1

# Determine total number of cells
total_cells = (A + 1) * (B + 1)

# Calculate minimum and maximum values
if N < total_cells:
    min_val = 0
else:
    min_val = min(counts.values()) if counts else 0
max_val = max(counts.values()) if counts else 0

# Output the result
print(min_val, max_val)