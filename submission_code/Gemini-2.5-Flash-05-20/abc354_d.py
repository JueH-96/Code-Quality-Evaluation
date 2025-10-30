# YOUR CODE HERE
import sys

def get_black_area_from_origin(X, Y):
    """
    Calculates the number of black 1x1 cells in the rectangle [0, X) x [0, Y).
    A cell [i, i+1) x [j, j+1) is black if (i + j) is even.
    """
    if X <= 0 or Y <= 0:
        return 0

    # Number of even integers in [0, N-1]
    # For N=1, evens: [0]. count = 1. (1+1)//2 = 1.
    # For N=2, evens: [0, 1]. count = 1. (2+1)//2 = 1.
    # For N=3, evens: [0, 1, 2]. count = 2. (3+1)//2 = 2.
    num_even_in_X = (X + 1) // 2
    num_odd_in_X = X // 2 # Number of odd integers in [0, N-1]

    num_even_in_Y = (Y + 1) // 2
    num_odd_in_Y = Y // 2

    # Black cells are where (i+j) is even, meaning:
    # (i is even AND j is even) OR (i is odd AND j is odd)
    black_cells_count = num_even_in_X * num_even_in_Y + num_odd_in_X * num_odd_in_Y
    return black_cells_count

def solve():
    A, B, C, D = map(int, sys.stdin.readline().split())

    # Calculate total black area using inclusion-exclusion principle.
    # Area([A,C) x [B,D)) = Area([0,C) x [0,D)) - Area([0,A) x [0,D)) - Area([0,C) x [0,B)) + Area([0,A) x [0,B))
    
    total_black_area = (
        get_black_area_from_origin(C, D)
        - get_black_area_from_origin(A, D)
        - get_black_area_from_origin(C, B)
        + get_black_area_from_origin(A, B)
    )

    # The problem asks for twice the area.
    print(2 * total_black_area)

if __name__ == '__main__':
    solve()