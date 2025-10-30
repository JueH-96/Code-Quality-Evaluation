import sys

def solve():
    # Read input from stdin
    line = sys.stdin.readline().split()
    A = int(line[0])
    B = int(line[1])
    D = int(line[2])

    # Generate the arithmetic sequence
    # An arithmetic sequence starts at A and increments by D.
    # We need to generate terms until we reach B.
    # Since the problem guarantees that B is a term reachable from A with difference D,
    # we can use a loop that goes from A up to B, incrementing by D.

    # Using a for loop with range is efficient.
    # range(start, stop, step) generates numbers from start up to (but not including) stop,
    # with increments of step.
    # Since B is the last term, the sequence includes B.
    # The next term after B would be B + D.
    # So, range(A, B + D, D) will generate A, A+D, A+2D, ..., B.
    sequence_terms = list(range(A, B + D, D))

    # Print the terms separated by spaces.
    # The '*' operator unpacks the list elements as arguments to the print function.
    # The print function, when given multiple arguments, prints them separated by spaces by default.
    print(*sequence_terms)

# Call the solve function to run the program
solve()