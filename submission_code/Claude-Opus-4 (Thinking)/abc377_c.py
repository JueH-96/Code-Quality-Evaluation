# YOUR CODE HERE
N, M = map(int, input().split())

occupied = set()
captured = set()

# Knight moves: 2 in one direction, 1 perpendicular
knight_moves = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

for _ in range(M):
    a, b = map(int, input().split())
    occupied.add((a, b))
    
    # Add all squares this piece can capture
    for di, dj in knight_moves:
        ni, nj = a + di, b + dj
        if 1 <= ni <= N and 1 <= nj <= N:
            captured.add((ni, nj))

# Squares we cannot place our piece on
unavailable = occupied | captured

# Total squares - unavailable squares
answer = N * N - len(unavailable)
print(answer)