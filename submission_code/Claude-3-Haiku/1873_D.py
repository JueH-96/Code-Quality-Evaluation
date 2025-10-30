from typing import List

def min_operations(n: int, k: int, s: str) -> int:
    """
    Finds the minimum number of operations needed to remove all black cells from the paper strip.

    Args:
        n (int): The length of the paper strip.
        k (int): The number of consecutive cells that can be made white in an operation.
        s (str): The string representing the paper strip, where 'B' represents a black cell and 'W' represents a white cell.

    Returns:
        int: The minimum number of operations needed to remove all black cells.
    """
    black_cells = [i for i, c in enumerate(s) if c == 'B']
    if not black_cells:
        return 0

    operations = 0
    i = 0
    while i < len(black_cells):
        start = black_cells[i]
        end = start + k - 1
        while i < len(black_cells) and black_cells[i] <= end:
            i += 1
        operations += 1

    return operations