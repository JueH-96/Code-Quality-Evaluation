# YOUR CODE HERE
N, M = map(int, input().split())

# Store existing piece positions
existing_pieces = set()
for _ in range(M):
    a, b = map(int, input().split())
    existing_pieces.add((a, b))

# Knight move offsets
knight_moves = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

# Find all squares under attack
under_attack = set()

for a, b in existing_pieces:
    for di, dj in knight_moves:
        ni, nj = a + di, b + dj
        # Check if the new position is within bounds
        if 1 <= ni <= N and 1 <= nj <= N:
            under_attack.add((ni, nj))

# Remove squares that already have pieces (they're not available anyway)
under_attack = under_attack - existing_pieces

# Total squares = N^2
# Empty squares = N^2 - M (occupied squares)
# Safe empty squares = Empty squares - squares under attack that are empty
total_squares = N * N
occupied_squares = M
empty_squares = total_squares - occupied_squares
unsafe_empty_squares = len(under_attack)

safe_squares = empty_squares - unsafe_empty_squares

print(safe_squares)