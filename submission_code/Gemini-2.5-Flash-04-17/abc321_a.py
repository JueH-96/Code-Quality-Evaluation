# YOUR CODE HERE
# Read the integer input N from stdin.
N = int(input())

# Convert the integer N to a string to easily access its digits by index.
s = str(N)

# Assume the number is 321-like initially.
# This flag will be set to False if we find any violation of the strictly decreasing condition.
is_strictly_decreasing = True

# Iterate through the string of digits from the first character (index 0)
# up to the second-to-last character (index len(s)-2).
# For each index i, we compare the digit at s[i] with the next digit s[i+1].
# The loop range `len(s) - 1` results in indices from 0 up to len(s) - 2.
# If s is a single digit (len(s) is 1), range(1-1) is range(0), which is an empty sequence.
# The loop will not execute, and the flag `is_strictly_decreasing` will remain True,
# which is correct for all single-digit numbers.
for i in range(len(s) - 1):
    # The condition for a 321-like number is that each digit must be
    # strictly greater than the digit that follows it.
    # If we find any instance where the digit s[i] is NOT strictly greater
    # than the next digit s[i+1] (i.e., s[i] <= s[i+1]), then the number
    # does not satisfy the condition and is not a 321-like number.
    # We can compare the digit characters directly because their standard
    # encoding (like ASCII or Unicode) is ordered numerically ('0' < '1' < ... < '9').
    if s[i] <= s[i+1]:
        is_strictly_decreasing = False
        # Once we find a pair of digits that violates the condition,
        # we know the entire number is not 321-like, so we can stop checking
        # the remaining pairs.
        break

# After checking all necessary pairs of digits (or if the number had only one digit),
# we check the value of the flag to determine the output.
if is_strictly_decreasing:
    # If the flag is still True, it means no violation was found, so the number is 321-like.
    print("Yes")
else:
    # If the flag was set to False, it means a violation was found, so the number is not 321-like.
    print("No")