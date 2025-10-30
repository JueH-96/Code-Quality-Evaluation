import sys

def can_color_grid(N, M, cells):
    # Initialize variables to keep track of the constraints
    row_black = [False] * (N + 1)
    col_black = [False] * (N + 1)
    row_white = [False] * (N + 1)
    col_white = [False] * (N + 1)

    for x, y, c in cells:
        if c == 'B':
            if y > 1 and col_white[y-1]:
                return "No"
            row_black[x] = True
            col_black[y] = True
        else:
            if y > 1 and col_black[y-1]:
                return "No"
            row_white[x] = True
            col_white[y] = True

    for i in range(1, N + 1):
        if row_black[i] and row_white[i]:
            return "No"
        if col_black[i] and col_white[i]:
            return "No"

    return "Yes"

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    cells = []

    index = 2
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index + 1])
        C = data[index + 2]
        cells.append((X, Y, C))
        index += 3

    result = can_color_grid(N, M, cells)
    print(result)

if __name__ == "__main__":
    main()