# YOUR CODE HERE

import sys

def main():
    n = int(sys.stdin.readline())
    sheets = []
    for _ in range(n):
        a, b, c, d = map(int, sys.stdin.readline().split())
        sheets.append((a, b, c, d))

    x_coords = set()
    y_coords = set()
    for a, b, c, d in sheets:
        x_coords.add(a)
        x_coords.add(b)
        y_coords.add(c)
        y_coords.add(d)

    x_coords = sorted(list(x_coords))
    y_coords = sorted(list(y_coords))

    area = 0
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            for a, b, c, d in sheets:
                if a <= x_coords[i] < x_coords[i + 1] and c <= y_coords[j] < y_coords[j + 1]:
                    area += (x_coords[i + 1] - x_coords[i]) * (y_coords[j + 1] - y_coords[j])
                    break

    print(area)

if __name__ == '__main__':
    main()