import sys

# Read input from standard input
N = int(sys.stdin.readline())

# Determine the approximation based on the magnitude of N
if N < 1000:
    # If N is less than or equal to 10^3-1 (i.e., N <= 999), print N as it is.
    result = N
elif N < 10000:
    # If N is between 10^3 and 10^4-1 (i.e., 1000 <= N <= 9999),
    # truncate the ones digit (divide by 10, take floor, multiply by 10).
    result = (N // 10) * 10
elif N < 100000:
    # If N is between 10^4 and 10^5-1 (i.e., 10000 <= N <= 99999),
    # truncate the tens digit and all digits below it (divide by 100, floor, multiply by 100).
    result = (N // 100) * 100
elif N < 1000000:
    # If N is between 10^5 and 10^6-1 (i.e., 100000 <= N <= 999999),
    # truncate the hundreds digit and all digits below it (divide by 1000, floor, multiply by 1000).
    result = (N // 1000) * 1000
elif N < 10000000:
    # If N is between 10^6 and 10^7-1 (i.e., 1000000 <= N <= 9999999),
    # truncate the thousands digit and all digits below it (divide by 10000, floor, multiply by 10000).
    result = (N // 10000) * 10000
elif N < 100000000:
    # If N is between 10^7 and 10^8-1 (i.e., 10000000 <= N <= 99999999),
    # truncate the ten-thousands digit and all digits below it (divide by 100000, floor, multiply by 100000).
    result = (N // 100000) * 100000
elif N < 1000000000:
    # If N is between 10^8 and 10^9-1 (i.e., 100000000 <= N <= 999999999),
    # truncate the hundred-thousands digit and all digits below it (divide by 1000000, floor, multiply by 1000000).
    result = (N // 1000000) * 1000000
# According to the constraints, N is between 0 and 10^9 - 1 inclusive,
# so N will always fall into one of the above conditional blocks.

# Print the calculated result to standard output
print(result)