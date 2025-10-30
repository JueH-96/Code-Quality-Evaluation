import sys

def find_nth_good_integer(N):
    # Define the base as 5 since there are 5 even digits (0, 2, 4, 6, 8)
    base = 5

    # Initialize the result and current power
    result = 0
    current_power = 1

    # Decrement N by 1 to handle 0-based indexing
    N -= 1

    # Loop until N is reduced to 0
    while N > 0:
        # Find the current digit by taking N modulo base
        current_digit = N % base
        # Add the current digit multiplied by the current power to the result
        result += current_digit * current_power
        # Update N by performing integer division by base
        N //= base
        # Multiply the current power by 10 for the next digit position
        current_power *= 10

    # Return the result as a string, replacing digits 3 and 4 with 2 and 8 respectively
    return int(str(result).replace('3', '2').replace('4', '8'))

# Read input from stdin
input = sys.stdin.read().strip()
N = int(input)

# Find the N-th smallest good integer
result = find_nth_good_integer(N)

# Write the output to stdout
sys.stdout.write(str(result) + '
')