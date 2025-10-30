import sys

def min_operations(n, k, s):
    """
    Calculate the minimum number of operations needed to remove all black cells.

    Args:
    n (int): The length of the paper.
    k (int): The number of consecutive cells that can be made white in one operation.
    s (str): A string of length n consisting of characters 'B' (representing a black cell) or 'W' (representing a white cell).

    Returns:
    int: The minimum number of operations needed to remove all black cells.
    """
    operations = 0
    i = 0
    while i < n:
        # If the current cell is black, perform an operation to make k consecutive cells white
        if s[i] == 'B':
            operations += 1
            # Move the pointer to the next cell after the operation
            i += k
        else:
            # If the current cell is white, move to the next cell
            i += 1
    return operations

def main():
    # Read the number of test cases
    t = int(sys.stdin.readline().strip())
    for _ in range(t):
        # Read the length of the paper and the number of consecutive cells that can be made white in one operation
        n, k = map(int, sys.stdin.readline().strip().split())
        # Read the string representing the paper
        s = sys.stdin.readline().strip()
        # Calculate and print the minimum number of operations needed to remove all black cells
        print(min_operations(n, k, s))

if __name__ == "__main__":
    main()