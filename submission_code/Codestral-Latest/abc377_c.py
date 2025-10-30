import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

pieces = []
for i in range(M):
    a = int(data[2 + 2 * i])
    b = int(data[3 + 2 * i])
    pieces.append((a, b))

# Directions a knight can move
directions = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

# Set to keep track of squares that can be captured
captured = set()

# Function to check if a square is within the grid
def is_valid(x, y):
    return 1 <= x <= N and 1 <= y <= N

# Mark all squares that can be captured by existing pieces
for a, b in pieces:
    for dx, dy in directions:
        nx, ny = a + dx, b + dy
        if is_valid(nx, ny):
            captured.add((nx, ny))

# Total number of squares
total_squares = N * N

# Number of squares that are either occupied or captured
occupied_or_captured = M + len(captured)

# Number of empty squares where you can place your piece
safe_squares = total_squares - occupied_or_captured

print(safe_squares)