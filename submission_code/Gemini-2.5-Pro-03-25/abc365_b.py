# YOUR CODE HERE
import sys

def solve():
    # Read the number of elements N from standard input
    n = int(sys.stdin.readline())
    # Read the sequence A of N integers from standard input
    a = list(map(int, sys.stdin.readline().split()))

    # Create a list of tuples, where each tuple contains an element from A
    # and its original 1-based index.
    # We store the 1-based index (i + 1) directly to avoid conversion later.
    indexed_a = []
    for i in range(n):
        # Append the tuple (value, 1-based index) to the list
        indexed_a.append((a[i], i + 1))

    # Sort the list of tuples based on the value (the first element of the tuple)
    # in descending order (from largest to smallest).
    # The `key=lambda x: x[0]` argument tells `sorted` to use the first element
    # of each tuple (the value) for comparison.
    # The `reverse=True` argument specifies descending order.
    sorted_indexed_a = sorted(indexed_a, key=lambda x: x[0], reverse=True)

    # After sorting, the first element `sorted_indexed_a[0]` corresponds to the
    # largest value, and the second element `sorted_indexed_a[1]` corresponds to
    # the second largest value.
    # We need the original index of the second largest value. This index is stored
    # as the second element (index 1) of the tuple `sorted_indexed_a[1]`.
    second_largest_index_1_based = sorted_indexed_a[1][1]

    # Print the 1-based index of the second largest element to standard output.
    print(second_largest_index_1_based)

# Call the solve function to run the program logic
solve()