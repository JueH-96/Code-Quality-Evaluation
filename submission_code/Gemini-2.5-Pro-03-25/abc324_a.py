# YOUR CODE HERE
import sys

def solve():
    # Read the integer N from the first line of input
    n = int(sys.stdin.readline())

    # Read the N integers A_1, A_2, ..., A_N from the second line of input
    # Split the line by spaces and convert each part to an integer
    a = list(map(int, sys.stdin.readline().split()))

    # Check if all elements in the list 'a' are equal.
    # One way to do this is to convert the list to a set.
    # A set only stores unique elements. If all elements in the list
    # are the same, the resulting set will have only one element.
    unique_elements = set(a)

    # If the number of unique elements is 1, it means all elements were equal.
    if len(unique_elements) == 1:
        print("Yes")
    else:
        # Otherwise, there were at least two different values.
        print("No")

# Call the solve function to execute the logic
solve()