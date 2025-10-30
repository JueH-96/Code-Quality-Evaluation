# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    W = int(input[idx + 1])
    M = int(input[idx + 2])
    idx += 3

    row_colors = [0] * H
    col_colors = [0] * W
    row_painted = [False] * H
    col_painted = [False] * W

    for _ in range(M):
        T = int(input[idx])
        A = int(input[idx + 1]) - 1
        X = int(input[idx + 2])
        idx += 3
        if T == 1:
            if not row_painted[A]:
                row_colors[A] = X
                row_painted[A] = True
            else:
                row_colors[A] = X
        elif T == 2:
            if not col_painted[A]:
                col_colors[A] = X
                col_painted[A] = True
            else:
                col_colors[A] = X

    color_count = defaultdict(int)
    for r in range(H):
        for c in range(W):
            if row_painted[r]:
                color_count[row_colors[r]] += 1
            elif col_painted[c]:
                color_count[col_colors[c]] += 1
            else:
                color_count[0] += 1

    result = sorted(color_count.items())
    print(len(result))
    for color, count in result:
        print(color, count)

if __name__ == "__main__":
    main()