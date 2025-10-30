# YOUR CODE HERE
import sys

def solve():
    # Read N, M, P from a single line of standard input
    # map(int, ...) converts the space-separated strings to integers
    N, M, P = map(int, sys.stdin.readline().split())

    count = 0
    
    # Initialize current_day to the first full moon day, M.
    current_day = M

    # Loop as long as the current full moon day is within the inclusive range [1, N].
    # Since M >= 1 and P >= 1, current_day will always be >= 1 once initialized.
    while current_day <= N:
        # If current_day is within the range, it's a valid full moon day.
        count += 1
        # Move to the next full moon day by adding P.
        current_day += P
    
    # Print the final count to standard output, followed by a newline.
    sys.stdout.write(str(count) + "
")

# Call the solve function to execute the program
solve()