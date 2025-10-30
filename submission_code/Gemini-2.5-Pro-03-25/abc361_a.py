# YOUR CODE HERE
import sys

def solve():
    # Read N, K, X from the first line
    n, k, x = map(int, sys.stdin.readline().split())

    # Read the sequence A from the second line
    a = list(map(int, sys.stdin.readline().split()))

    # The problem statement uses 1-based indexing for K.
    # Python lists use 0-based indexing.
    # The K-th element in the sequence A (1-based) is at index K-1 in the list a (0-based).
    # We want to insert X *immediately after* the K-th element.
    # This means we should insert X at index K (0-based).
    # For example, if K=3, we insert after the element at index 2, meaning the new element X
    # will be at index 3.
    # Python's list.insert(index, value) inserts 'value' *before* the element currently at 'index'.
    # So, inserting at index K achieves the desired result.

    # Insert X into the list 'a' at index K
    a.insert(k, x)

    # Print the elements of the modified list 'a', separated by spaces.
    # The * operator unpacks the list elements as separate arguments to print.
    # print by default separates arguments with a space.
    print(*a)

# Call the solve function to execute the logic
solve()

# END OF YOUR CODE