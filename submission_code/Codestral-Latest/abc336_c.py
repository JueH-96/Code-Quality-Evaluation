# YOUR CODE HERE
import sys
from itertools import product

def find_nth_good_integer(N):
    if N == 1:
        return 0

    # Generate all possible good integers with up to 12 digits
    good_integers = []
    for length in range(1, 13):
        for combo in product('02468', repeat=length):
            if combo[0] != '0':  # Ensure the number is not leading with zero
                good_integers.append(int(''.join(combo)))

    # Sort the good integers
    good_integers.sort()

    # The N-th smallest good integer
    return good_integers[N-1]

# Read input
N = int(sys.stdin.read().strip())

# Find and print the N-th smallest good integer
print(find_nth_good_integer(N))