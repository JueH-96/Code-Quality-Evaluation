# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input from stdin, solves the problem, and prints the answer to stdout.
    """
    # Read N and K from the first line of standard input.
    # N is the length of the sequence A, and K is the upper bound of the range.
    # The value of N is not explicitly needed in this solution's logic but is
    # part of the problem's input specification.
    try:
        N, K = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handles empty input lines, e.g., at the end of a file.
        return

    # Read the sequence A of N integers from the second line.
    # We use map() to create an efficient iterator of integers.
    A = map(int, sys.stdin.readline().split())

    # Calculate the sum of all integers from 1 to K using the arithmetic series formula.
    # Sum = n * (first + last) / 2 => K * (1 + K) / 2.
    # Python's arbitrary-precision integers handle large values of K without overflow.
    sum_1_to_K = K * (K + 1) // 2

    # To find the sum of integers in [1, K] that are *not* in A, we can subtract
    # the sum of unique integers in A that are also in [1, K] from the total sum.
    # A set comprehension is a Pythonic and efficient way to build the set of
    # unique elements from A that are within the desired range.
    # Since the problem states A contains positive integers, we only need to check x <= K.
    unique_A_in_range = {x for x in A if x <= K}

    # Calculate the sum of these unique elements from A.
    sum_to_subtract = sum(unique_A_in_range)

    # The final result is the difference between the total sum and the sum of elements to exclude.
    result = sum_1_to_K - sum_to_subtract

    # Print the final answer.
    print(result)

# Call the solve function to run the program.
solve()