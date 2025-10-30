# YOUR CODE HERE
import sys

# Read the input from stdin
X = int(sys.stdin.readline())

# Initialize N and current factorial
# Since X >= 2, the smallest positive integer N such that N! >= X is at least 2.
# We can start calculating factorials from N=2.
N = 2
current_factorial = 2 # 2!

# Calculate factorials iteratively
# We stop when current_factorial is no longer less than X
# Since X is guaranteed to be the factorial of some positive integer N,
# current_factorial must eventually equal X.
# The loop calculates 3!, 4!, ..., N! until N! == X
while current_factorial < X:
    N += 1
    current_factorial *= N

# At this point, current_factorial is equal to X, and N is the corresponding integer.
# Print the answer to stdout
print(N)