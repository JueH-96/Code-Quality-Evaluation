# YOUR CODE HERE
import sys

# Read input from stdin
# N: base, M: exponent limit
n, m = map(int, sys.stdin.readline().split())

# Define the upper limit for the sum
limit = 10**9

# Initialize sum and the current term (starts with N^0 = 1)
current_sum = 0
term = 1

# Flag to indicate if the sum exceeds the limit
is_inf = False

# The sum is X = N^0 + N^1 + ... + N^M. There are M+1 terms.
# Iterate through the exponents from 0 to M.
for i in range(m + 1):
    # At the start of iteration i, 'term' holds the value of N^i.

    # Check 1: Prevent overflow when adding the current term to the sum.
    # If current_sum + term would exceed the limit (10^9), then the final sum X will also exceed the limit.
    # We check this as term > limit - current_sum. This form is safer for conceptual limits.
    # If term is already greater than limit, term > limit - current_sum will be true (since current_sum >= 0).
    if term > limit - current_sum:
        is_inf = True
        break # Sum will exceed limit, no need to continue

    # Add the current term (N^i) to the sum.
    current_sum += term

    # Check 2: If this is not the last term (i < M), prepare the next term (N^(i+1)) for the next iteration.
    # Before calculating the next term, check if the next term itself would exceed the limit.
    # If the next term N^(i+1) > limit, then the total sum will definitely be > limit.
    # This is relevant only when N > 1, as the term grows.
    if i < m:
        # The next term is N^(i+1) = N^i * N = term * n.
        if n > 1:
            # If term * n > limit, we can stop early.
            # For positive n, this is equivalent to checking if term > limit / n.
            # Using integer division for positive n: if term > limit // n, then term * n > limit.
            # This check allows early exit if terms grow very rapidly.
            # This check is valid because term currently holds N^i, and we are predicting N^(i+1).
            if term > limit // n:
                 is_inf = True
                 break # The next term N^(i+1) is too large

            # Calculate the next term N^(i+1) for the next iteration.
            term *= n
        # If N == 1, term remains 1, so no update is needed in this branch.


# After the loop completes, check the is_inf flag.
if is_inf:
    # If the sum exceeded the limit at any point, print "inf".
    print("inf")
else:
    # Otherwise, print the calculated sum.
    print(current_sum)