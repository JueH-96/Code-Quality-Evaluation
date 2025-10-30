import sys

def solve():
    """
    Reads an integer N from standard input and prints a level-N carpet.
    
    The program generates the pattern by determining the color of each cell (i, j)
    individually. The rule, derived from analyzing the sample outputs, is based on
    the base-3 representation of the coordinates. A cell (i, j) is white ('.')
    if and only if there exists a digit position 'k' where the k-th digit of both
    i and j in their base-3 forms is 1. Otherwise, the cell is black ('#').
    
    This method is chosen as it correctly reproduces all sample cases, whereas a
    direct implementation of the recursive description in the problem statement
    does not match the provided sample for N=2.
    """
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Constraints guarantee valid input, but this is good practice.
        return

    # The size of the grid is 3^N x 3^N.
    size = 3**N

    # Iterate over each row of the grid.
    for i in range(size):
        row_chars = []
        # Iterate over each column in the current row.
        for j in range(size):
            is_white = False
            temp_i, temp_j = i, j
            
            # Check the base-3 digits of i and j.
            # A number less than 3^N has N digits in base 3 (with leading zeros).
            for _ in range(N):
                # If the current digit (remainder of division by 3) is 1 for both...
                if temp_i % 3 == 1 and temp_j % 3 == 1:
                    # ...the cell is white.
                    is_white = True
                    break
                
                # Move to the next digit to the left (integer division by 3).
                temp_i //= 3
                temp_j //= 3
            
            if is_white:
                row_chars.append('.')
            else:
                row_chars.append('#')
        
        # Join the characters and print the row to standard output.
        sys.stdout.write("".join(row_chars) + '
')

solve()