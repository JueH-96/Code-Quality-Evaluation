# YOUR CODE HERE
import sys

def solve():
    # Read the input line from standard input
    # input() reads a line from stdin and removes the trailing newline
    # .split() splits the line into a list of strings based on whitespace
    line_parts = sys.stdin.readline().split()

    # The first part is the surname S
    S = line_parts[0]
    # The second part is the first name T (which is not used in the output)
    # T = line_parts[1]

    # Construct the output string by concatenating the surname S,
    # a space " ", and the honorific "san"
    # Using an f-string for cleaner formatting
    output_string = f"{S} san"

    # Print the final output string to standard output
    print(output_string)

# Call the solve function to execute the logic
solve()