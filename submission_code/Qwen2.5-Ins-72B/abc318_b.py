# YOUR CODE HERE
import sys

def area_of_union(rectangles):
    if not rectangles:
        return 0

    # Initialize the grid
    grid = [[0] * 101 for _ in range(101)]

    # Mark the grid for each rectangle
    for A, B, C, D in rectangles:
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = 1

    # Calculate the area of the union
    area = sum(sum(row) for row in grid)
    return area

# Read input
N = int(input())
rectangles = [list(map(int, input().split())) for _ in range(N)]

# Compute and print the result
print(area_of_union(rectangles))