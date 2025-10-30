import sys

# Read the input integer X from standard input.
X = int(sys.stdin.readline())

# Initialize N and its factorial.
# We start with N=1, where 1! = 1.
# The loop will then calculate 2!, 3!, and so on, until it finds X.
n = 1
current_factorial = 1

# Iterate to find the positive integer N such that N! equals X.
# The problem guarantees that such an N exists and is unique,
# and that X is not less than 2. This implies N must be at least 2.
while True:
    # Check if the current factorial value matches X.
    if current_factorial == X:
        # If it matches, N is found. Print N and exit the loop.
        print(n)
        break
    
    # If the current factorial does not match X,
    # increment N and calculate the next factorial value.
    n += 1
    current_factorial *= n