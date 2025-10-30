import sys

# Read N from standard input
N = int(sys.stdin.readline())

# Function to check if a number is a "326-like" number
def is_326_like(num):
    """
    A 326-like number is a three-digit positive integer where
    the product of the hundreds and tens digits equals the ones digit.
    """
    
    # Extract the hundreds digit
    hundreds_digit = num // 100
    
    # Extract the tens digit
    tens_digit = (num // 10) % 10
    
    # Extract the ones digit
    ones_digit = num % 10
    
    # Check the condition: product of hundreds and tens equals ones
    return hundreds_digit * tens_digit == ones_digit

# Start searching from N upwards
# The constraints guarantee that N is between 100 and 919.
# The largest possible 326-like number is 919 (9 * 1 = 9).
# Therefore, the loop will always find an answer within a small range.
current_num = N
while True:
    if is_326_like(current_num):
        # If the current number is 326-like, it's the smallest one
        # greater than or equal to N, so print it and terminate.
        sys.stdout.write(str(current_num) + "
")
        break
    # If not, check the next integer
    current_num += 1