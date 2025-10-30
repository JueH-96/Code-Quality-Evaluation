import bisect
from collections import defaultdict

# Read input
w, h = map(int, input().split())
n = int(input())
strawberries = [tuple(map(int, input().split())) for _ in range(n)]
A = int(input())
a = list(map(int, input().split()))
B = int(input())
b = list(map(int, input().split()))

# Initialize the count dictionary
counts = defaultdict(int)

# Process each strawberry
for p, q in strawberries:
    # Find the vertical strip
    v = bisect.bisect_left(a, p)
    # Find the horizontal strip
    h_idx = bisect.bisect_left(b, q)
    # Update the count
    counts[(v, h_idx)] += 1

# Calculate the maximum count
max_count = max(counts.values()) if counts else 0

# Calculate the minimum count
total_pairs = (A + 1) * (B + 1)
if total_pairs <= n:
    min_count = min(counts.values())
else:
    min_count = 0

# Output the results
print(min_count, max_count)