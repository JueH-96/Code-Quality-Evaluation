# YOUR CODE HERE
def count_black_cells(A, B, C, D):
    def count_cells(x1, y1, x2, y2):
        # Count the number of cells in the rectangle (x1, y1) to (x2, y2)
        return (x2 - x1) * (y2 - y1)

    def count_black_cells_in_rectangle(A, B, C, D):
        # Calculate the number of black cells in the rectangle (A, B) to (C, D)
        total_cells = count_cells(A, B, C, D)
        even_x_lines = (C - A) // 2
        even_y_lines = ((D - B) // 2) if B % 2 == 0 else ((D - B - 1) // 2)
        even_diagonal_lines = ((C - A + D - B) // 2) if (A + B) % 2 == 0 else ((C - A + D - B - 1) // 2)
        
        # Total number of lines that divide the rectangle
        total_lines = even_x_lines + even_y_lines + even_diagonal_lines
        
        # The number of black cells is half the total cells plus or minus the number of lines
        # depending on the starting color of the bottom-left cell
        if (A + B) % 2 == 0:
            black_cells = (total_cells + total_lines) // 2
        else:
            black_cells = (total_cells - total_lines) // 2
        
        return black_cells

    black_cells = count_black_cells_in_rectangle(A, B, C, D)
    return 2 * black_cells

import sys
input = sys.stdin.read().strip().split()
A, B, C, D = map(int, input)
print(count_black_cells(A, B, C, D))