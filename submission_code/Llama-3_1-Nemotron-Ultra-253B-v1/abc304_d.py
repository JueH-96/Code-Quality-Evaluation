import bisect
from collections import defaultdict

# Read input
w, h = map(int, input().split())
n = int(input())
strawberries = [tuple(map(int, input().split())) for _ in range(n)]
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Calculate frequency of each (v, h) group
freq = defaultdict(int)
for p, q in strawberries:
    v = bisect.bisect_left(a, p)
    h = bisect.bisect_left(b, q)
    freq[(v, h)] += 1

# Compute total pieces and determine min and max
A = len(a)
B = len(b)
total_pieces = (A + 1) * (B + 1)
max_count = max(freq.values()) if freq else 0

if len(freq) < total_pieces:
    min_count = 0
else:
    min_count = min(freq.values())

print(min_count, max_count)