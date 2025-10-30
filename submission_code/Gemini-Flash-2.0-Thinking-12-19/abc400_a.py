# Read the input integer A
A = int(input())

# The total number of people is 400.
# In a rectangular formation of A rows and B columns, the total number of people is A * B.
# So, we have the equation A * B = 400.
# We need to find a positive integer B that satisfies this equation for the given A.
# This means B must be 400 / A.
# For B to be a positive integer, A must be a divisor of 400, and A must be positive (which is guaranteed by the constraints 1 <= A <= 400).

# Check if 400 is divisible by A
if 400 % A == 0:
    # If 400 is divisible by A, then B = 400 / A is a positive integer.
    # Calculate B using integer division (although regular division would also work since we know it's divisible)
    B = 400 // A
    # Print the value of B
    print(B)
else:
    # If 400 is not divisible by A, there is no positive integer B that satisfies the condition.
    # Print -1
    print(-1)