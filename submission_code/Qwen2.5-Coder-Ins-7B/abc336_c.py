# YOUR CODE HERE
import sys

def nth_good_integer(N):
    if N == 1:
        return 0
    N -= 1
    digits = []
    while N > 0:
        digits.append(N % 5)
        N //= 5
    good_integer = 0
    for digit in digits:
        good_integer = good_integer * 10 + digit * 2
    return good_integer

# Read input
N = int(sys.stdin.read().strip())

# Output the result
print(nth_good_integer(N))