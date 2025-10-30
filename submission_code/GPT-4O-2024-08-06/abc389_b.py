# YOUR CODE HERE
import sys

def find_factorial_number(X):
    # Start with N = 1 and factorial = 1
    N = 1
    factorial = 1
    
    # Increment N and calculate factorial until it matches X
    while factorial < X:
        N += 1
        factorial *= N
    
    # Since it's guaranteed that there is exactly one such N, we can return it
    return N

# Read input
X = int(sys.stdin.read().strip())

# Find and print the result
print(find_factorial_number(X))