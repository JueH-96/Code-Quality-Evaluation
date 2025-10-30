import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    max_b_y = defaultdict(int)  # row -> max B's column
    min_w_y = defaultdict(lambda: N + 1)  # row -> min W's column
    max_b_x = defaultdict(int)  # column -> max B's row
    min_w_x = defaultdict(lambda: N + 1)  # column -> min W's row
    w_cells = []

    for _ in range(M):
        X, Y, C = sys.stdin.readline().split()
        X = int(X)
        Y = int(Y)
        # Process row
        if C == 'B':
            if Y > max_b_y[X]:
                max_b_y[X] = Y
        else:
            if Y < min_w_y[X]:
                min_w_y[X] = Y
            w_cells.append((X, Y))
        # Process column
        if C == 'B':
            if X > max_b_x[Y]:
                max_b_x[Y] = X
        else:
            if X < min_w_x[Y]:
                min_w_x[Y] = X

    # Check all rows
    for r in range(1, N + 1):
        low_r = max_b_y.get(r, 0)
        min_w = min_w_y.get(r, N + 1)
        high_r = min_w - 1
        if low_r > high_r:
            print("No")
            return

    # Check all columns
    for c in range(1, N + 1):
        low_c = max_b_x.get(c, 0)
        min_w = min_w_x.get(c, N + 1)
        high_c = min_w - 1
        if low_c > high_c:
            print("No")
            return

    # Check W cells
    for X, Y in w_cells:
        low_r = max_b_y.get(X, 0)
        if Y <= low_r:
            low_c = max_b_x.get(Y, 0)
            if low_c >= X:
                print("No")
                return

    print("Yes")

if __name__ == "__main__":
    main()