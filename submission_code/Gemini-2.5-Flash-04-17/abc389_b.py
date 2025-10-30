# YOUR CODE HERE
import sys

# Read the input integer X from standard input.
# X is guaranteed to be an integer such that 2 <= X <= 3 * 10^18.
X = int(sys.stdin.readline())

# Initialize the current factorial value.
# We start with the value of 1! which is 1.
# This variable `current_factorial` will iteratively store the factorial values (1!, 2!, 3!, ...)
# for the current value of N being tested.
current_factorial = 1

# Initialize N. We are looking for a positive integer N.
# We will iterate through N = 1, 2, 3, ...
# We start the counter N from 0, and increment it at the beginning of the loop
# so that the first value of N we calculate the factorial for is 1 (when N becomes 1).
N = 0

# Loop indefinitely to calculate factorials iteratively until the factorial value matches X.
# The loop is guaranteed to terminate because the problem statement guarantees that X is
# exactly the factorial of some unique positive integer N.
while True:
    # Increment N to the next positive integer we want to test.
    # In the first iteration, N becomes 1. In the second, N becomes 2, and so on.
    N = N + 1

    # Calculate the factorial for the current value of N.
    # The factorial of N (N!) is equal to the factorial of (N-1), i.e., (N-1)!, multiplied by N.
    # `current_factorial` currently stores the factorial of (N-1).
    current_factorial = current_factorial * N

    # Check if the calculated factorial for the current N matches the input value X.
    if current_factorial == X:
        # If the calculated factorial matches X, we have found the required integer N.
        # Print the value of N to standard output as the answer.
        print(N)
        # Exit the loop because we are guaranteed that there is exactly one such N,
        # and we have found it.
        break

    # Due to the problem guarantee that X is exactly the factorial of some positive integer N,
    # we are certain to find a match (current_factorial == X) eventually.
    # Thus, the loop is guaranteed to terminate correctly via the 'break' statement.
    # An explicit check to see if `current_factorial` has exceeded X
    # is not strictly necessary, as the equality condition will be met first based on the guarantee.