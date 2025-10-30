import sys

def is_palindrome(num):
    """
    Checks if a number's decimal representation is a palindrome.

    Args:
        num (int): The number to check.

    Returns:
        bool: True if the number is a palindrome, False otherwise.
    """
    s = str(num)
    return s == s[::-1]

def solve():
    """
    Finds the maximum palindromic cube number not greater than N.
    Reads N from standard input and prints the result to standard output.
    """
    N = int(sys.stdin.readline())

    # Initialize max_palindromic_cube.
    # The problem states N is a positive integer and K (palindromic cube) must be positive.
    # The smallest positive cube is 1^3 = 1. Its string representation "1" is a palindrome.
    # Thus, 1 is always a valid palindromic cube for any N >= 1.
    # We initialize with 0, and it will be updated to at least 1.
    max_palindromic_cube = 0

    # We are searching for a positive integer K such that K = x^3 and K <= N.
    # Given N <= 10^18, we can determine the maximum possible value for x.
    # If K = x^3 <= 10^18, then x <= (10^18)^(1/3).
    # (10^18)^(1/3) = 10^(18/3) = 10^6.
    # So, x will not exceed 1,000,000.
    # We iterate x from 1 up to 1,000,000 (inclusive).
    for x in range(1, 1000001):
        K = x * x * x # Calculate the cube of x

        # Optimization: If the current cube K exceeds N, then any larger x
        # will also produce a cube greater than N (since x is increasing, K is strictly increasing).
        # Therefore, we can stop the search.
        if K > N:
            break

        # Check if the calculated cube K is a palindrome.
        if is_palindrome(K):
            # Since we iterate x in increasing order, the value of K (x^3) is also increasing.
            # If K is a palindrome and K <= N, it becomes the largest palindromic cube found so far
            # that meets the criteria.
            max_palindromic_cube = K

    # Print the final result (the maximum palindromic cube found).
    sys.stdout.write(str(max_palindromic_cube) + "
")

# Call the solve function to execute the program logic.
solve()