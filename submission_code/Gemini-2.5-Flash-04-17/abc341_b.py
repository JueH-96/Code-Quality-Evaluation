# YOUR CODE HERE
import sys

# Read the input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))
S = []
T = []
for _ in range(N - 1):
    s, t = map(int, sys.stdin.readline().split())
    S.append(s)
    T.append(t)

# current_currency_amount will track the total available amount
# of the currency *before* applying the current operation's conversion.
# Start with the initial amount of currency 1 (A[0]).
# In iteration `op_idx` (from 0 to N-2), current_currency_amount
# holds the total available amount of currency `op_idx + 1` that can be
# used for conversion to currency `op_idx + 2`.
# We use this to calculate the total available amount of currency `op_idx + 2`
# which becomes the new current_currency_amount for the next iteration.
current_currency_amount = A[0]

# Loop through operations i = 1 to N-1 (using 0-indexed S and T arrays).
# Operation index `op_idx` (0 to N-2) corresponds to operation `op_idx + 1`.
# This operation converts currency `op_idx + 1` to `op_idx + 2`.
for op_idx in range(N - 1):
    # At the start of this iteration `op_idx`, current_currency_amount holds
    # the total amount of currency `op_idx + 1` available from initial A[op_idx]
    # and conversions from currency op_idx.

    # Calculate how many times operation `op_idx + 1` can be performed
    # using the available currency `op_idx + 1`.
    # Each operation costs S[op_idx] units of currency `op_idx + 1`.
    if S[op_idx] > 0: # Constraint says S_i >= 1, so S[op_idx] is always > 0
        num_operations = current_currency_amount // S[op_idx]
    else:
        # This case should not happen based on constraints S_i >= 1
        # If it could happen, division by zero is possible.
        # Assuming S_i >= 1 based on constraints.
        pass # num_operations calculation is safe

    # Calculate the amount of currency `op_idx + 2` produced.
    # Each operation yields T[op_idx] units of currency `op_idx + 2`.
    produced_amount = num_operations * T[op_idx]

    # The total amount of currency `op_idx + 2` available for the next step
    # (or as the final amount if this is the last iteration)
    # is the initial amount A[op_idx + 1] plus the amount produced.
    # This total amount becomes the new current_currency_amount for the next iteration
    # (which processes currency `op_idx + 2` -> `op_idx + 3`).
    current_currency_amount = A[op_idx + 1] + produced_amount

# After the loop finishes (when op_idx = N-2),
# current_currency_amount holds the total available amount of currency (N-2) + 2 = N.
print(current_currency_amount)