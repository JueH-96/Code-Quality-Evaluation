# YOUR CODE HERE
import bisect

# Read cake dimensions
W, H = map(int, input().split())

# Read strawberries
N = int(input())
strawberries = []
for _ in range(N):
    p, q = map(int, input().split())
    strawberries.append((p, q))

# Read vertical cuts (x coordinates)
A = int(input())
if A > 0:
    a_cuts = list(map(int, input().split()))
else:
    a_cuts = []

# Read horizontal cuts (y coordinates)
B = int(input())
if B > 0:
    b_cuts = list(map(int, input().split()))
else:
    b_cuts = []

# Add boundaries
x_boundaries = [0] + a_cuts + [W]
y_boundaries = [0] + b_cuts + [H]

# Count strawberries in each piece
piece_counts = {}

for p, q in strawberries:
    # Find which piece this strawberry belongs to
    # Use binary search to find the interval
    x_idx = bisect.bisect_right(a_cuts, p)
    y_idx = bisect.bisect_right(b_cuts, q)
    
    # Count this strawberry in the corresponding piece
    if (x_idx, y_idx) not in piece_counts:
        piece_counts[(x_idx, y_idx)] = 0
    piece_counts[(x_idx, y_idx)] += 1

# Find min and max
if piece_counts:
    min_count = min(piece_counts.values())
    max_count = max(piece_counts.values())
else:
    min_count = 0
    max_count = 0

# Some pieces might have 0 strawberries
total_pieces = (A + 1) * (B + 1)
if len(piece_counts) < total_pieces:
    min_count = 0

print(min_count, max_count)