# YOUR CODE HERE
def main():
    """
    Solves the problem by discretizing the plane onto a grid.
    """
    # Read the number of rectangles.
    try:
        N = int(input())
    except (EOFError, ValueError):
        return

    # The problem states that coordinates are between 0 and 100.
    # Specifically, for a rectangle defined by (A, B, C, D), we have:
    # 0 <= A < B <= 100
    # 0 <= C < D <= 100
    # A unit square with its bottom-left corner at (i, j) covers the region
    # i <= x < i+1 and j <= y < j+1.
    # To cover the rectangle's x-range [A, B], we need to mark unit squares
    # with x-coordinates from A to B-1. The maximum possible value for B is 100,
    # so the maximum x-index accessed will be 99. The same logic applies to y.
    # Therefore, a 100x100 grid is sufficient.
    
    plane = [[0 for _ in range(100)] for _ in range(100)]

    # Process each rectangle.
    for _ in range(N):
        A, B, C, D = map(int, input().split())
        
        # Mark all 1x1 unit squares within this rectangle's bounds.
        # The loops iterate from A to B-1 for the x-coordinate and C to D-1
        # for the y-coordinate.
        for x_coord in range(A, B):
            for y_coord in range(C, D):
                # Setting the cell to 1 marks it as covered. If already 1, it stays 1.
                plane[x_coord][y_coord] = 1

    # The total area is the count of unique 1x1 squares that have been marked.
    # Since marked squares are 1 and unmarked are 0, this is the sum of the grid.
    total_area = sum(sum(row) for row in plane)

    print(total_area)

if __name__ == "__main__":
    main()