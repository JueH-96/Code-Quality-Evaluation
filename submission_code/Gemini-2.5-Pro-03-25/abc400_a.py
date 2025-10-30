# YOUR CODE HERE
import sys

# Read the integer A from standard input
a_str = sys.stdin.readline()
a = int(a_str)

# Define the total number of people
total_people = 400

# Check if A is a divisor of total_people
# A must be a positive integer (given by constraint 1 <= A <= 400)
# B must also be a positive integer.
# If A * B = 400, then B = 400 / A.
# B will be an integer if and only if 400 is divisible by A.
# B will be positive since 400 and A are positive.
if total_people % a == 0:
    # Calculate B using integer division
    b = total_people // a
    # Print the value of B
    print(b)
else:
    # If A is not a divisor of 400, no such integer B exists
    print("-1")