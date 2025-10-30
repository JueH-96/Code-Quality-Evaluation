# YOUR CODE HERE
import sys

# Read the integer B from standard input.
# The input B is guaranteed to be an integer between 1 and 10^18.
B = int(sys.stdin.readline())

# Initialize the result variable to -1. This value will be printed if no positive integer A
# satisfying A^A = B is found.
result_A = -1

# We are looking for a positive integer A such that A^A = B.
# Let's examine the function f(A) = A^A for positive integers A:
# f(1) = 1^1 = 1
# f(2) = 2^2 = 4
# f(3) = 3^3 = 27
# ...
# The function f(A) = A^A is strictly increasing for A >= 1.
# We need to determine the maximum possible value for A given that B <= 10^18.
# Let's compute A^A for increasing values of A:
# 15^15 = 437,893,890,380,859,375 (which is less than 10^18)
# 16^16 = 18,446,744,073,709,551,616 (which is greater than 10^18)
# Since 16^16 exceeds the maximum possible value of B (10^18),
# any potential integer solution A must be less than 16.
# Therefore, we only need to check integer values of A from 1 to 15.

# Iterate through possible integer values of A starting from 1 up to (but not including) 16.
for A in range(1, 16):
    # Calculate A raised to the power of A.
    # Python's built-in integers support arbitrary precision, so calculations
    # involving potentially large numbers like 15^15 are handled correctly without overflow.
    power_val = A ** A

    # Compare the calculated value A^A with the input B.
    if power_val == B:
        # If A^A equals B, we have found the required positive integer A.
        # Store A in the result variable.
        result_A = A
        # Since A^A is strictly increasing for A >= 1, there can be at most one solution.
        # We can stop searching once the solution is found.
        break
    
    # Optimization: If A^A becomes greater than B, we can stop searching early.
    # Because the function A^A is strictly increasing for A >= 1,
    # if A^A is already larger than B, then for any A' > A, (A')^(A') will also be larger than B.
    # Thus, no larger value of A can be the solution.
    if power_val > B:
        break

# Print the final result to standard output.
# If a solution was found (i.e., the loop found an A such that A^A = B), result_A holds the value of A.
# If the loop completed without finding such an A (either by checking all values up to 15
# or by breaking early due to the optimization), result_A remains at its initial value of -1.
print(result_A)

# END OF YOUR CODE HERE