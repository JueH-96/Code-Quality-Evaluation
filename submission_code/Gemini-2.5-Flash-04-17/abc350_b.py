# YOUR CODE HERE
import sys

# Read N and Q
line1 = sys.stdin.readline().split()
n = int(line1[0])
q = int(line1[1])

# Read the treatment list T
treatments = []
# Check if Q > 0 because if Q is 0, the second line might not be present in the input.
if q > 0:
    treatments_line = sys.stdin.readline().split()
    # Convert treatment strings to integers
    treatments = [int(t) for t in treatments_line]

# Use a list to count the number of times each hole is treated.
# The list size is N + 1 to use 1-based indexing, matching hole numbers 1 to N.
# Index 0 is unused. Initialize all counts to zero.
treatment_counts = [0] * (n + 1)

# Iterate through the list of treatments and increment the count for each treated hole.
for t in treatments:
    # The problem constraints guarantee 1 <= T_i <= N, so t is a valid index.
    treatment_counts[t] += 1

# After all treatments are counted, determine the final state of each hole.
# Initially, each hole (1 to N) has a tooth.
# Each treatment on a hole toggles the presence of a tooth.
# A hole treated an even number of times (including 0) will end up with its initial state (a tooth).
# A hole treated an odd number of times will end up with the opposite of its initial state (no tooth).
# We need to count the number of holes (1 to N) that have a tooth at the end.
# This is the number of holes whose treatment count is even.
final_teeth_count = 0
for i in range(1, n + 1):
    if treatment_counts[i] % 2 == 0:
        # This hole ends up with a tooth because it was treated an even number of times
        # starting from a state with a tooth.
        final_teeth_count += 1

# Print the final number of teeth.
print(final_teeth_count)