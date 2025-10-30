# YOUR CODE HERE
import sys

# The problem asks to read from standard input and write to standard output.
# The iterative approach used here does not require increasing the recursion depth.

def solve():
    """
    Reads the initial sequence from stdin, performs the specified insertion procedure
    until the termination condition is met, and prints the final sequence to stdout.
    """
    # Read N, the initial length of the sequence.
    # While N is given in the input, the algorithm operates on the list A
    # and its length changes during the procedure. We read N to adhere to
    # the input format, but the primary loop logic uses len(A).
    N = int(sys.stdin.readline())
    
    # Read the initial sequence A as a list of integers from the second line of input.
    # The map(int, ...) converts the space-separated string numbers into integers.
    A = list(map(int, sys.stdin.readline().split()))

    # The main loop implementing the insertion procedure.
    # It continues iterating as long as insertions are being made.
    while True:
        # This flag is set to True if we find and process a gap in the current pass.
        # If a full pass completes without setting this flag, the procedure terminates.
        found_gap = False
        
        # Initialize the index 'i' to start scanning from the beginning of the list A.
        i = 0
        
        # Inner loop to iterate through the sequence A to find the first adjacent pair
        # (A[i], A[i+1]) whose absolute difference is not equal to 1.
        # The loop condition `i < len(A) - 1` ensures we always have a valid A[i+1].
        while i < len(A) - 1:
            # Check the absolute difference between the current adjacent terms.
            if abs(A[i] - A[i+1]) != 1:
                # If the absolute difference is not 1, we have found a gap that needs filling.
                # This is the *first* such gap encountered in this scan because we iterate from i=0.
                found_gap = True

                # Determine the sequence of numbers to be inserted between A[i] and A[i+1].
                insertion_list = []
                if A[i] < A[i+1]:
                    # If A[i] is smaller, insert numbers strictly between A[i] and A[i+1]
                    # in ascending order: A[i]+1, A[i]+2, ..., A[i+1]-1.
                    # The range(start, stop) function generates integers from 'start' up to 'stop - 1'.
                    # So, range(A[i] + 1, A[i+1]) correctly generates the required sequence.
                    insertion_list = list(range(A[i] + 1, A[i+1]))
                else: # A[i] > A[i+1]
                    # If A[i] is greater than A[i+1], insert numbers strictly between A[i] and A[i+1]
                    # in descending order: A[i]-1, A[i]-2, ..., A[i+1]+1.
                    # The range(start, stop, step) function generates integers starting from 'start',
                    # decreasing by 'step' (which is -1 here), until 'stop' is reached (exclusive).
                    # To get numbers down to A[i+1]+1, the 'stop' value should be A[i+1].
                    # E.g., range(5-1, 1, -1) -> range(4, 1, -1) generates [4, 3, 2] (stops before 1).
                    # E.g., range(5-1, 2, -1) -> range(4, 2, -1) generates [4, 3] (stops before 2).
                    insertion_list = list(range(A[i] - 1, A[i+1], -1))

                # Insert the list of generated numbers into the sequence A at index i+1.
                # This efficiently places the elements of insertion_list between the original A[i] and A[i+1].
                # Slice assignment A[i+1:i+1] = list_to_insert is the standard Pythonic way to do this.
                A[i+1:i+1] = insertion_list

                # According to the procedure description ("Return to step 1"), after finding and
                # processing the first gap, we should restart the process from the beginning
                # on the modified sequence. The outer `while True` loop handles this restart.
                # We break the inner loop here because we've found and processed the first gap
                # for this iteration of the outer loop. The outer loop will then check `found_gap`
                # and continue, starting a new scan from the beginning (`i=0`).
                break # Exit the inner loop scan for the current outer loop iteration

            # If the absolute difference between A[i] and A[i+1] was exactly 1, this pair is correctly formed.
            # Move to the next adjacent pair in the sequence to continue the scan.
            i += 1

        # After the inner loop finishes its scan (either because it found a gap and broke,
        # or because it iterated through all adjacent pairs without finding a gap):
        # If the `found_gap` flag is still False, it means the inner loop completed
        # without finding any pair (A[i], A[i+1]) where `abs(A[i] - A[i+1]) != 1`.
        # This state (all adjacent differences are 1) is the termination condition
        # for the entire procedure.
        if not found_gap:
            # Break the outer loop to end the procedure.
            break

    # The procedure has terminated. The list A now holds the final sequence.
    # Print the elements of the final sequence to standard output, separated by spaces.
    # The `*A` syntax unpacks the list A into individual arguments for the print function.
    # The default separator for print with multiple arguments is a space.
    print(*A)

# Call the main solve function to execute the program logic.
solve()