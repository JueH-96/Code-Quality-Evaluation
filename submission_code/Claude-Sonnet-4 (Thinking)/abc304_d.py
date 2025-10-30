import bisect

W, H = map(int, input().split())
N = int(input())
strawberries = []
for _ in range(N):
    p, q = map(int, input().split())
    strawberries.append((p, q))

A = int(input())
a = list(map(int, input().split()))

B = int(input())
b = list(map(int, input().split()))

# Count strawberries in each piece
piece_counts = {}

for p, q in strawberries:
    # Find which x-segment contains p
    x_idx = bisect.bisect_right(a, p)
    
    # Find which y-segment contains q
    y_idx = bisect.bisect_right(b, q)
    
    piece = (x_idx, y_idx)
    piece_counts[piece] = piece_counts.get(piece, 0) + 1

# Get all possible counts
total_pieces = (A + 1) * (B + 1)
pieces_with_strawberries = len(piece_counts)

counts = list(piece_counts.values())
max_count = max(counts)

if pieces_with_strawberries < total_pieces:
    # Some pieces have 0 strawberries
    min_count = 0
else:
    # All pieces have at least one strawberry
    min_count = min(counts)

print(min_count, max_count)