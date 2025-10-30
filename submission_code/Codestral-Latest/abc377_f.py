# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
m = int(data[1])
pieces = [tuple(map(int, data[i].split())) for i in range(2, 2 + m)]

rows = set()
cols = set()
diag1 = set()
diag2 = set()

for a, b in pieces:
    rows.add(a)
    cols.add(b)
    diag1.add(a + b)
    diag2.add(a - b)

total_squares = n * n
captured_squares = len(rows) * n + len(cols) * n - len(rows) * len(cols) + len(diag1) * n + len(diag2) * n - len(diag1) * len(diag2)

# Subtract the squares that are counted multiple times
for a, b in pieces:
    if a in rows and b in cols:
        captured_squares -= n - 1
    if a + b in diag1 and a - b in diag2:
        captured_squares -= n - 1

# Ensure we don't subtract too much
captured_squares = min(captured_squares, total_squares)

safe_squares = total_squares - captured_squares
print(safe_squares)