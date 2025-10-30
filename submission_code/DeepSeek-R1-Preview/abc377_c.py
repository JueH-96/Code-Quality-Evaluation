# Read input
N, M = map(int, input().split())
existing_pieces = set()

for _ in range(M):
    a, b = map(int, input().split())
    existing_pieces.add((a, b))

# Directions for knight moves
directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

dangerous = set()

for (a, b) in existing_pieces:
    for dx, dy in directions:
        x = a + dx
        y = b + dy
        if 1 <= x <= N and 1 <= y <= N:
            dangerous.add((x, y))

# Calculate forbidden squares
forbidden = existing_pieces.union(dangerous)
answer = N * N - len(forbidden)

print(answer)