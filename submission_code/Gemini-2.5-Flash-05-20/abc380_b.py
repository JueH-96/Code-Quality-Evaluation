# YOUR CODE HERE
import sys

def solve():
    # Read the input string S from standard input
    # .strip() removes any leading/trailing whitespace, including the newline character
    S = sys.stdin.readline().strip()

    # Split the string S by the '|' character.
    # For example, "|---|-|" will split into ['', '---', '-', ''].
    # The first and last empty strings are due to S starting and ending with '|'.
    parts = S.split('|')

    # The actual sequences of '-' characters are contained in the elements
    # from index 1 up to the second-to-last element.
    # We slice the list 'parts' to get only these relevant segments.
    # For example, ['', '---', '-', ''] becomes ['---', '-'].
    relevant_segments = parts[1:-1]

    # For each segment, its length tells us the corresponding A_i value.
    # We use a list comprehension to calculate the length of each segment.
    A_sequence = [len(segment) for segment in relevant_segments]

    # Print the elements of the A_sequence, separated by spaces.
    # The '*' operator unpacks the list into arguments for the print function.
    print(*A_sequence)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()