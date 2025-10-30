# YOUR CODE HERE
import sys

# Read the input integer N
N = int(sys.stdin.readline())

# Repeatedly divide N by 2 as long as it is divisible by 2
while N % 2 == 0:
    N //= 2

# Repeatedly divide N by 3 as long as it is divisible by 3
while N % 3 == 0:
    N //= 3

# If the remaining N is 1, it means the original N was composed solely of prime factors 2 and 3.
# This corresponds to N = 2^x * 3^y for some non-negative integers x and y.
if N == 1:
    print("Yes")
else:
    # If N is not 1, it means there was at least one prime factor other than 2 or 3.
    print("No")