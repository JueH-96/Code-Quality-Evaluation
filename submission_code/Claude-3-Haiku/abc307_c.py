# Read input
h_a, w_a = map(int, input().split())
a = [input() for _ in range(h_a)]
h_b, w_b = map(int, input().split())
b = [input() for _ in range(h_b)]
h_x, w_x = map(int, input().split())
x = [input() for _ in range(h_x)]

# Check if Takahashi can achieve his goal
def can_achieve_goal():
    # Create a set of black squares in A and B
    black_squares = set()
    for i in range(h_a):
        for j in range(w_a):
            if a[i][j] == '#':
                black_squares.add((i, j))
    for i in range(h_b):
        for j in range(w_b):
            if b[i][j] == '#':
                black_squares.add((i, j))

    # Check if all black squares in A and B are included in the cut-out sheet
    for i in range(h_x):
        for j in range(w_x):
            if x[i][j] == '#' and (i, j) not in black_squares:
                return False

    # Check if the cut-out sheet matches sheet X
    for i in range(h_x):
        for j in range(w_x):
            if x[i][j] != '.' and a[i][j] != '#' and b[i][j] != '#':
                return False

    return True

if can_achieve_goal():
    print("Yes")
else:
    print("No")