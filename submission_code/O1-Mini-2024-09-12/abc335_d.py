# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.read())
    grid = [[0 for _ in range(N)] for _ in range(N)]
    mid = N // 2
    grid[mid][mid] = 'T'
    top, bottom = 0, N-1
    left, right = 0, N-1
    number = 1
    while top <= bottom and left <= right:
        # Traverse left to right on top row
        for j in range(left, right+1):
            if not (top == mid and j == mid):
                grid[top][j] = number
                number +=1
        top +=1
        # Traverse top to bottom on right column
        for i in range(top, bottom+1):
            if not (i == mid and right == mid):
                grid[i][right] = number
                number +=1
        right -=1
        # Traverse right to left on bottom row
        if top <= bottom:
            for j in range(right, left-1, -1):
                if not (bottom == mid and j == mid):
                    grid[bottom][j] = number
                    number +=1
            bottom -=1
        # Traverse bottom to top on left column
        if left <= right:
            for i in range(bottom, top-1, -1):
                if not (i == mid and left == mid):
                    grid[i][left] = number
                    number +=1
            left +=1
    for row in grid:
        print(' '.join(str(cell) for cell in row))

if __name__ == "__main__":
    main()