# YOUR CODE HERE
import sys

def solve():
    """
    Reads input N and sequence A, determines if A is strictly increasing,
    and prints "Yes" or "No".
    """
    # Read N from the first line
    n = int(sys.stdin.readline())

    # Read the sequence A from the second line, splitting by space and converting to integers
    a = list(map(int, sys.stdin.readline().split()))

    # Assume the sequence is strictly increasing initially
    is_strictly_increasing = True

    # Iterate through the sequence from the first element up to the second-to-last element
    # We need to compare element i with element i+1
    for i in range(n - 1):
        # Check if the current element is greater than or equal to the next element
        # If it is, the sequence is not strictly increasing
        if a[i] >= a[i+1]:
            is_strictly_increasing = False
            # We found a violation, no need to check further
            break

    # Print the result based on the flag
    if is_strictly_increasing:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the logic
solve()