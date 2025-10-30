# YOUR CODE HERE
import sys

def solve():
    A, B = map(int, sys.stdin.readline().split())

    # Use a set to store unique x values found
    possible_x_values = set()

    # Case 1: A, B, x forms an arithmetic sequence (B is the middle term)
    # 2 * B = A + x  =>  x = 2 * B - A
    x1 = 2 * B - A
    possible_x_values.add(x1)

    # Case 2: B, A, x forms an arithmetic sequence (A is the middle term)
    # 2 * A = B + x  =>  x = 2 * A - B
    x2 = 2 * A - B
    possible_x_values.add(x2)

    # Case 3: A, x, B forms an arithmetic sequence (x is the middle term)
    # 2 * x = A + B
    # For x to be an integer, A + B must be an even number.
    if (A + B) % 2 == 0:
        x3 = (A + B) // 2  # Use integer division
        possible_x_values.add(x3)

    # The number of distinct integers x is the size of the set
    print(len(possible_x_values))

# Call the solve function to run the program
solve()