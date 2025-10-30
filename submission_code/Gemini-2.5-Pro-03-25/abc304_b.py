# YOUR CODE HERE
import sys

# Read the integer N from standard input
# Use strip() to remove potential leading/trailing whitespace including newline
n_str = sys.stdin.readline().strip() 
# Convert the input string to an integer
n = int(n_str)

# Determine the approximation based on the value of N using if/elif conditions
# The conditions check the range of N based on powers of 10.

if n < 1000: # If N is less than 1000 (i.e., 0 <= N <= 999)
    # As per the rule: If N is less than or equal to 10^3-1, print N as it is.
    print(n)
elif n < 10000: # If N is between 1000 and 9999 (inclusive)
    # As per the rule: If N is between 10^3 and 10^4-1, truncate the ones digit.
    # This is achieved by integer division by 10, then multiplying by 10.
    # Example: 1234 -> 1234 // 10 * 10 = 123 * 10 = 1230.
    divisor = 10
    print(n // divisor * divisor)
elif n < 100000: # If N is between 10000 and 99999 (inclusive)
    # As per the rule: If N is between 10^4 and 10^5-1, truncate the tens digit and below.
    # Integer division by 100, then multiply by 100.
    # Example: 12345 -> 12345 // 100 * 100 = 123 * 100 = 12300.
    divisor = 100
    print(n // divisor * divisor)
elif n < 1000000: # If N is between 100000 and 999999 (inclusive)
    # As per the rule: If N is between 10^5 and 10^6-1, truncate the hundreds digit and below.
    # Integer division by 1000, then multiply by 1000.
    # Example: 500600 -> 500600 // 1000 * 1000 = 500 * 1000 = 500000.
    divisor = 1000
    print(n // divisor * divisor)
elif n < 10000000: # If N is between 1000000 and 9999999 (inclusive)
    # As per the rule: If N is between 10^6 and 10^7-1, truncate the thousands digit and below.
    # Integer division by 10000, then multiply by 10000.
    # Example: 1234567 -> 1234567 // 10000 * 10000 = 123 * 10000 = 1230000.
    divisor = 10000
    print(n // divisor * divisor)
elif n < 100000000: # If N is between 10000000 and 99999999 (inclusive)
    # As per the rule: If N is between 10^7 and 10^8-1, truncate the ten-thousands digit and below.
    # Integer division by 100000, then multiply by 100000.
    # Example: 20230603 -> 20230603 // 100000 * 100000 = 202 * 100000 = 20200000.
    divisor = 100000
    print(n // divisor * divisor)
else: # If N is between 100000000 and 999999999 (inclusive)
    # As per the rule: If N is between 10^8 and 10^9-1, truncate the hundred-thousands digit and below.
    # Integer division by 1000000, then multiply by 1000000.
    # Example: 123456789 -> 123456789 // 1000000 * 1000000 = 123 * 1000000 = 123000000.
    divisor = 1000000
    print(n // divisor * divisor)