import sys

def solve():
    """
    Reads an integer N and finds the smallest 326-like number greater than or equal to N.
    """
    try:
        # Read the integer N from standard input.
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handle potential empty input or invalid format, though not expected by constraints.
        return

    # Start checking from N upwards.
    current_num = N
    while True:
        # Convert the number to a string to easily access its digits.
        s_num = str(current_num)
        
        # Extract the hundreds, tens, and ones digits as integers.
        hundreds_digit = int(s_num[0])
        tens_digit = int(s_num[1])
        ones_digit = int(s_num[2])
        
        # Check if it's a 326-like number: product of first two digits equals the third.
        if hundreds_digit * tens_digit == ones_digit:
            # Found the smallest 326-like number >= N. Print it and exit.
            print(current_num)
            break
        
        # If not, increment the number and check the next one.
        current_num += 1

solve()