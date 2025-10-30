import sys

# Read the input integer X
X = int(sys.stdin.readline())

# Initialize N and the current factorial value
# We start with N=1, so 1! = 1.
# Since X >= 2, we know the answer N must be at least 2.
# The loop will calculate 2!, 3!, 4!, ... until it matches X.
n = 1
current_factorial = 1

# Iterate through positive integers n = 2, 3, 4, ...
# calculating n! until we find the one that equals X.
# The problem guarantees that such an N exists and is unique.
while True:
    # Move to the next integer N (from 1 to 2, then 2 to 3, etc.)
    n += 1
    
    # Calculate the factorial for the new N: N! = (N-1)! * N
    # Python's integers handle large numbers automatically, so we don't
    # need to worry about overflow within the given constraints.
    current_factorial *= n
    
    # Check if the calculated factorial matches the input X
    if current_factorial == X:
        # We found the integer n such that n! = X
        # Print the value of n and exit the program.
        print(n)
        break # Exit the while loop