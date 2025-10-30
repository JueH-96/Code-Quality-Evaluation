import sys

# Read input
N, M = map(int, sys.stdin.readline().split())

# The limit for the sum
Limit = 10**9

# Initialize the sum and the current term (which starts as N^0)
sum_X = 0
term = 1 # Represents N^i, starts with N^0 = 1

# Iterate through the terms from N^0 up to N^M
# There are M+1 terms in total (for i = 0, 1, ..., M)
for i in range(M + 1):
    # Add the current term (N^i) to the sum
    # 'term' currently holds the value of N^i for this iteration
    sum_X = sum_X + term

    # Check if the sum has exceeded the limit *after* adding the current term
    if sum_X > Limit:
        print("inf")
        sys.exit() # The sum is already too large, no need to continue

    # Prepare the term for the *next* iteration.
    # The next term is N^(i+1) = N^i * N.
    # 'term' currently holds N^i. We update it to N^(i+1) for the next step.
    # This calculation is only needed if there is a next iteration (i < M).
    if i < M:
        # Calculate the next term N^(i+1)
        term = term * N

# If the loop finishes without sum_X exceeding the limit, print the final calculated sum
print(sum_X)