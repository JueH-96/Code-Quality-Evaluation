import sys

def find_missing_cookie(H, W, grid):
    # Initialize variables to track the boundaries of the rectangle
    top, bottom = H, 0
    left, right = W, 0

    # Find the boundaries of the rectangle
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                top = min(top, i)
                bottom = max(bottom, i)
                left = min(left, j)
                right = max(right, j)

    # Check each square within the rectangle to find the missing cookie
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':
                return i + 1, j + 1

    return -1, -1  # This line should never be reached given the problem constraints

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    grid = [data[i] for i in range(2, 2 + H)]

    i, j = find_missing_cookie(H, W, grid)
    print(i, j)

if __name__ == "__main__":
    main()