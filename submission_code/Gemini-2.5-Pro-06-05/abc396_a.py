# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input, solves the problem, and prints the output.
    """
    try:
        # Read the integer N, which is the length of the sequence.
        N = int(sys.stdin.readline())
        
        # Read the sequence of N integers, A.
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle potential empty input lines or malformed input, though
        # problem constraints usually guarantee valid input.
        return

    # Iterate from the first element up to the third-to-last element.
    # The loop for index 'i' goes from 0 to N-3.
    # range(N - 2) generates integers from 0 up to N-3.
    for i in range(N - 2):
        # Check if three consecutive elements are identical using chained comparison.
        if A[i] == A[i+1] == A[i+2]:
            # If a triplet is found, print "Yes" and terminate the loop.
            print("Yes")
            break
    else:
        # The 'else' clause of a 'for' loop executes if the loop completes
        # without being terminated by a 'break' statement.
        # This means no such triplet was found.
        print("No")

solve()