N, M = map(int, input().split())
pieces = []
for _ in range(M):
    pieces.append(list(map(int, input().split())))

rows = set()
cols = set()
diag1 = set()
diag2 = set()

for piece in pieces:
    rows.add(piece[0])
    cols.add(piece[1])
    diag1.add(piece[0] + piece[1])
    diag2.add(piece[0] - piece[1])

total_squares = N * N
captured_squares = len(rows) * N + len(cols) * N - len(rows.intersection(cols)) * 1
captured_squares += len(diag1) * N - len(rows.intersection(diag1)) - len(cols.intersection(diag1))
captured_squares += len(diag2) * N - len(rows.intersection(diag2)) - len(cols.intersection(diag2))

for r in rows:
    for c in cols:
        if r + c in diag1 and r - c in diag2:
            captured_squares -= 1
            break

print(total_squares - captured_squares)