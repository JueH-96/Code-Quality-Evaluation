import sys

def solve():
    # Read the input string from standard input
    S = sys.stdin.readline().strip()

    # Split the string by the '|' character.
    # Since there are exactly two '|'s, this will produce a list of three parts:
    # [string_before_first_pipe, string_between_pipes, string_after_second_pipe]
    parts = S.split('|')

    # The desired result is the concatenation of the part before the first pipe
    # and the part after the second pipe.
    # parts[0] is the string before the first '|'.
    # parts[-1] is the string after the second '|'.
    result = parts[0] + parts[-1]

    # Print the resulting string to standard output
    print(result)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()