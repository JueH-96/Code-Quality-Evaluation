# YOUR CODE HERE
N, M = map(int, input().split())
pieces = []
occupied = set()

for _ in range(M):
    a, b = map(int, input().split())
    pieces.append((a, b))
    occupied.add((a, b))

# For each existing piece, find all positions that can capture it
unsafe = set()

# The 8 knight moves
moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

for a, b in pieces:
    # For each existing piece at (a,b), find positions (x,y) that can capture it
    # If (x,y) can capture (a,b), then (a,b) = (x+di, y+dj) for some move (di,dj)
    # So (x,y) = (a-di, b-dj)
    for di, dj in moves:
        x = a - di
        y = b - dj
        if 1 <= x <= N and 1 <= y <= N:
            unsafe.add((x, y))

# Total squares that cannot be used = occupied + unsafe (minus overlap)
cannot_use = occupied | unsafe

# Answer = total squares - squares that cannot be used
answer = N * N - len(cannot_use)
print(answer)