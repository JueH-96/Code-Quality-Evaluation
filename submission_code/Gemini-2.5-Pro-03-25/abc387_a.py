# YOUR CODE HERE
import sys

def solve():
    # Read the two integers A and B from standard input
    # The input is expected to be on a single line, space-separated
    line = sys.stdin.readline().split()
    A = int(line[0])
    B = int(line[1])

    # Calculate the sum of A and B
    total = A + B

    # Calculate the square of the sum
    result = total * total
    # Alternatively: result = total ** 2

    # Print the result to standard output
    print(result)

# Call the solve function to execute the logic
solve()