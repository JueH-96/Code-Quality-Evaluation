import bisect

W, H = map(int, input().split())
N = int(input())
strawberries = []
for _ in range(N):
    p, q = map(int, input().split())
    strawberries.append((p, q))

A = int(input())
a_values = list(map(int, input().split())) if A > 0 else []

B = int(input())  
b_values = list(map(int, input().split())) if B > 0 else []

# Create boundary lists including edges
x_bounds = [0] + a_values + [W]
y_bounds = [0] + b_values + [H]

# Count strawberries in each piece
piece_counts = {}

for p, q in strawberries:
    # Find which piece this strawberry belongs to
    i = bisect.bisect_right(x_bounds, p) - 1
    j = bisect.bisect_right(y_bounds, q) - 1
    piece_counts[(i, j)] = piece_counts.get((i, j), 0) + 1

if not piece_counts:
    print(0, 0)
else:
    max_count = max(piece_counts.values())
    total_pieces = (len(x_bounds) - 1) * (len(y_bounds) - 1)
    # If some pieces have no strawberries, min is 0
    min_count = 0 if len(piece_counts) < total_pieces else min(piece_counts.values())
    print(min_count, max_count)