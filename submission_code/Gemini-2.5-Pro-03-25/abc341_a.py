# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N from stdin, constructs the required alternating string,
    and prints it to stdout.
    """
    try:
        # Read the integer N from standard input
        n_str = sys.stdin.readline()
        if not n_str:
            # Handle empty input if necessary, though constraints suggest N >= 1
            return
        n = int(n_str.strip())

        # Validate constraints (optional based on problem statement, but good practice)
        if not (1 <= n <= 100):
            # Handle invalid N if necessary
            # For competitive programming, usually constraints are guaranteed
            pass

        # Construct the string: N pairs of "10" followed by a final "1"
        # This results in N zeros and (N + 1) ones, alternating, starting with 1.
        result_string = "10" * n + "1"

        # Print the resulting string to standard output
        print(result_string)

    except ValueError:
        # Handle cases where input is not a valid integer
        # Again, typically not needed if input format is guaranteed
        pass
    except Exception as e:
        # Catch any other unexpected errors
        # print(f"An error occurred: {e}", file=sys.stderr) # For debugging
        pass

# Call the solve function to execute the logic
if __name__ == "__main__":
    solve()