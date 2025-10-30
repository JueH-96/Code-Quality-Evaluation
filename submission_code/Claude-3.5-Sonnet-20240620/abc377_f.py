# YOUR CODE HERE
def can_capture(x1, y1, x2, y2):
    return (x1 == x2 or y1 == y2 or x1 + y1 == x2 + y2 or x1 - y1 == x2 - y2)

N, M = map(int, input().split())

rows = set()
cols = set()
diag1 = set()
diag2 = set()

for _ in range(M):
    a, b = map(int, input().split())
    rows.add(a)
    cols.add(b)
    diag1.add(a + b)
    diag2.add(a - b)

total_squares = N * N
captured = 0

for i in range(1, N + 1):
    for j in range(1, N + 1):
        if i in rows or j in cols or i + j in diag1 or i - j in diag2:
            captured += 1

safe_squares = total_squares - captured

print(safe_squares)