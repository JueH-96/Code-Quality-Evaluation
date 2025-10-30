import sys

# Read the integer N from standard input.
N = int(sys.stdin.readline())

# Determine the approximation based on N's value range.
if N < 10**3:
    # If N is less than 10^3 (i.e., N <= 999), print N as it is.
    result = N
elif N < 10**4:
    # If N is between 10^3 and 10^4-1 (inclusive), truncate the ones digit.
    # This is equivalent to dividing by 10 and then multiplying by 10.
    result = (N // 10) * 10
elif N < 10**5:
    # If N is between 10^4 and 10^5-1 (inclusive), truncate the tens digit and all digits below it.
    # This is equivalent to dividing by 100 and then multiplying by 100.
    result = (N // 100) * 100
elif N < 10**6:
    # If N is between 10^5 and 10^6-1 (inclusive), truncate the hundreds digit and all digits below it.
    # This is equivalent to dividing by 1000 and then multiplying by 1000.
    result = (N // 1000) * 1000
elif N < 10**7:
    # If N is between 10^6 and 10^7-1 (inclusive), truncate the thousands digit and all digits below it.
    # This is equivalent to dividing by 10000 and then multiplying by 10000.
    result = (N // 10000) * 10000
elif N < 10**8:
    # If N is between 10^7 and 10^8-1 (inclusive), truncate the ten-thousands digit and all digits below it.
    # This is equivalent to dividing by 100000 and then multiplying by 100000.
    result = (N // 100000) * 100000
else: # N >= 10^8 and N < 10^9 (since N is constrained to be less than 10^9)
    # If N is between 10^8 and 10^9-1 (inclusive), truncate the hundred-thousands digit and all digits below it.
    # This is equivalent to dividing by 1000000 and then multiplying by 1000000.
    result = (N // 1000000) * 1000000

# Print the calculated approximation.
print(result)