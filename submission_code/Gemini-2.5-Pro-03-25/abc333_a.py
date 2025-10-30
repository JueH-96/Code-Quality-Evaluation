import sys

# Read the integer N from standard input
n_str = sys.stdin.readline().strip()

# Convert the input string to an integer to get the repetition count
try:
    n_int = int(n_str)
except ValueError:
    # Handle potential errors if the input is not a valid integer, although constraints say it will be.
    # This part is defensive programming, might not be strictly needed given the constraints.
    print("Invalid input: Please enter an integer between 1 and 9.", file=sys.stderr)
    sys.exit(1)

# Validate the constraint N is between 1 and 9
if not (1 <= n_int <= 9):
    print(f"Input constraint violated: N must be between 1 and 9, inclusive. Got {n_int}", file=sys.stderr)
    sys.exit(1)

# The character to repeat is the same as the input string representation of N
char_to_repeat = n_str

# Repeat the character (string) n_int times using string multiplication
result_string = char_to_repeat * n_int

# Print the resulting string to standard output
print(result_string)