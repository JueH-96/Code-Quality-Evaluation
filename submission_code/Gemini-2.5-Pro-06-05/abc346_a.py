# YOUR CODE HERE
import sys

def solve():
    """
    This function reads the input, processes it according to the problem specification,
    and prints the output to standard out.
    """
    # Read the integer N from the first line of standard input.
    # While N is not strictly needed for the logic of this solution,
    # it is part of the specified input format.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # In a real-world scenario or a more complex problem,
        # robust error handling for input would be important.
        # For this problem, we assume valid input as per constraints.
        return

    # Read the space-separated integers from the second line into a list A.
    # The map() function applies int() to each element of the split string.
    try:
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        return

    # Use zip() to create an iterator of pairs of adjacent elements from A.
    # For a list A = [a0, a1, a2, ...], A[1:] is [a1, a2, ...].
    # zip(A, A[1:]) will yield (a0, a1), (a1, a2), and so on.
    adjacent_pairs = zip(A, A[1:])

    # A list comprehension provides a concise way to compute the product for each pair.
    B = [x * y for x, y in adjacent_pairs]

    # Print the elements of list B, separated by spaces.
    # The * operator unpacks the list B into positional arguments for the print function.
    # print(*B) is equivalent to print(B[0], B[1], ..., sep=' '),
    # which matches the required output format.
    print(*B)

solve()