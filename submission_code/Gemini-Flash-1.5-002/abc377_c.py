n, m = map(int, input().split())
occupied = set()
for _ in range(m):
    a, b = map(int, input().split())
    occupied.add((a, b))

total_squares = n * n
captured = set()
for r, c in occupied:
    for dr, dc in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
        nr, nc = r + dr, c + dc
        if 1 <= nr <= n and 1 <= nc <= n:
            captured.add((nr, nc))

print(total_squares - len(captured) - len(occupied))