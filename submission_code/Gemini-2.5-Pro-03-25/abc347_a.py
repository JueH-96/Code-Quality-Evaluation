# YOUR CODE HERE
import sys

def solve():
    # Read N and K from the first line of input
    # N: length of the sequence A
    # K: the divisor
    n, k = map(int, sys.stdin.readline().split())

    # Read the sequence A from the second line of input
    # A is a list of N integers
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize an empty list to store the quotients
    quotients = []

    # Iterate through each element 'x' in the sequence 'a'
    for x in a:
        # Check if the element 'x' is divisible by 'k'
        # The modulo operator (%) returns 0 if x is perfectly divisible by k
        if x % k == 0:
            # If 'x' is a multiple of 'k', calculate the quotient
            # using integer division '//'
            quotient = x // k
            # Add the calculated quotient to the 'quotients' list
            quotients.append(quotient)

    # The problem statement guarantees that the input sequence A is sorted
    # A_1 < A_2 < ... < A_N.
    # Since K is positive (K >= 1), dividing the multiples of K by K
    # will preserve their relative order.
    # Therefore, the 'quotients' list is already sorted in ascending order.

    # Print the elements of the 'quotients' list.
    # The * operator unpacks the list, passing each element as a separate
    # argument to the print function.
    # By default, the print function separates arguments with a space.
    print(*quotients)

# Call the solve function to execute the main logic
solve()