N, M = map(int, input().split())
pieces = []
for _ in range(M):
    a, b = map(int, input().split())
    pieces.append((a, b))

# Knight's moves (8 possible attacks)
moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

occupied = set(pieces)
attacked = set()

# Find all squares that can be attacked by existing pieces
for a, b in pieces:
    for di, dj in moves:
        ni, nj = a + di, b + dj
        if 1 <= ni <= N and 1 <= nj <= N:
            attacked.add((ni, nj))

# Safe squares = Total - Occupied - (Attacked but not occupied)
safe_count = N * N - len(occupied) - len(attacked - occupied)
print(safe_count)