# YOUR CODE HERE
import sys

# Read N, L, R from the first line
# N is the number of elements in A
# L and R define the inclusive range [L, R]
line1 = sys.stdin.readline().split()
N = int(line1[0])
L = int(line1[1])
R = int(line1[2])

# Read the sequence A from the second line
# Use sys.stdin.readline().split() and map(int, ...) to get a list of integers
A = list(map(int, sys.stdin.readline().split()))

# Initialize an empty list to store the calculated X_i values
X = []

# Iterate through each element 'a' in the sequence A
for a in A:
    # We need to find the value X_i in the range [L, R] that is closest to 'a'.
    # The value in [L, R] closest to 'a' is 'a' itself if 'a' falls within [L, R].
    # If 'a' is less than L, all values in [L, R] are greater than 'a'. The closest is L.
    # If 'a' is greater than R, all values in [L, R] are less than 'a'. The closest is R.
    
    if a < L:
        # If 'a' is strictly less than L, the minimum value in the range [L, R] is the closest.
        X_i = L
    elif a > R:
        # If 'a' is strictly greater than R, the maximum value in the range [L, R] is the closest.
        X_i = R
    else: # This condition implies L <= a <= R
        # If 'a' is within the inclusive range [L, R], 'a' itself is the closest value.
        X_i = a
        
    # Append the calculated X_i to the results list
    X.append(X_i)

# Print the elements of the results list X, separated by spaces.
# The print(*X) syntax unpacks the list elements as positional arguments to the print function.
# By default, print separates multiple arguments with a space.
print(*X)