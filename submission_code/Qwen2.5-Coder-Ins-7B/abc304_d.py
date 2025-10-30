# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

W = int(data[0])
H = int(data[1])
N = int(data[2])
strawberries = [(int(data[3 + 2 * i]), int(data[4 + 2 * i])) for i in range(N)]
A = int(data[3 + 2 * N])
A_lines = list(map(int, data[4 + 2 * N: 4 + 2 * N + A]))
B = int(data[4 + 2 * N + A])
B_lines = list(map(int, data[5 + 2 * N + A:]))

# Sort strawberries by x and y coordinates
strawberries.sort()

# Initialize a 2D list to count strawberries in each piece
pieces = [[0] * (B + 1) for _ in range(A + 1)]

# Count strawberries in each piece
for px, py in strawberries:
    x_index = next(i for i, a in enumerate(A_lines) if px < a)
    y_index = next(i for i, b in enumerate(B_lines) if py < b)
    pieces[x_index][y_index] += 1

# Find the minimum and maximum number of strawberries in any piece
min_strawberries = min(min(row) for row in pieces)
max_strawberries = max(max(row) for row in pieces)

print(min_strawberries, max_strawberries)