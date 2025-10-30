import sys

# Read input B from standard input
B = int(sys.stdin.readline())

# We are looking for a positive integer A such that A^A = B.
# Let's test small positive integer values for A starting from 1.
# The function f(A) = A^A grows very quickly for A >= 1.
# Given the constraint 1 <= B <= 10^18, A cannot be very large.
# Let's find the maximum A such that A^A <= 10^18.
# We can calculate A^A for small integer values of A:
# 1^1 = 1
# 2^2 = 4
# 3^3 = 27
# ...
# 10^10 = 10,000,000,000
# 11^11 = 285,311,670,611
# 12^12 = 8,916,100,448,256
# 13^13 = 302,875,106,592,253
# 14^14 = 758,335,243,623,346,304 (This value is less than or equal to 10^18)
# 15^15 = 4,378,938,903,808,593,750 (This value is greater than 10^18)
# This means if there exists a positive integer A such that A^A = B and B <= 10^18,
# then A must be a positive integer such that A^A <= 10^18.
# Based on the calculations, this implies A must be less than or equal to 14.
# Therefore, we only need to check positive integer values for A from 1 up to 14.

found_a = -1 # Initialize the result to -1, indicating no solution found yet.

# Loop through the possible values of A, from 1 to 14 (inclusive).
# range(1, 15) generates the integers 1, 2, 3, ..., 14.
for A in range(1, 15):
    # Calculate A raised to the power of A. Python handles arbitrarily large integers,
    # so this calculation is accurate for A <= 14.
    power_of_a = A ** A

    # Check if the calculated A^A is equal to the input B.
    if power_of_a == B:
        found_a = A # If they are equal, we found the solution A.
        break       # Exit the loop immediately as we only need one such A.

    # The function A^A is strictly increasing for A >= 1.
    # If the current A^A is already greater than B, then for any larger integer A',
    # A'^A' will also be greater than A^A and thus greater than B.
    # So, if power_of_a > B, we know no solution exists for this A or any larger A.
    # We can stop searching.
    # This check is useful if B is smaller than the maximum A^A in the loop range.
    if power_of_a > B:
        break # Exit the loop, as no solution will be found for A or any larger values.

# After the loop finishes (either by finding a solution and breaking, or by iterating through
# the range and potentially breaking early if A^A exceeds B), print the stored result.
# If no solution was found in the loop, found_a will remain -1.
print(found_a)