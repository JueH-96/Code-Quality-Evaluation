# Read the input integer A
A = int(input())

# The total number of people is 400.
# For a rectangular arrangement of A rows and B columns, the total people = A * B.
# So, we have the equation: A * B = 400.
# We need to find B. Therefore, B = 400 / A.

# For B to be a valid number of columns, it must be a positive integer.
# Since A is a positive integer (1 <= A <= 400), 400 / A will always be positive.
# The main condition to check is whether B is an integer, which means 400 must be
# perfectly divisible by A.

if 400 % A == 0:
    # If 400 is perfectly divisible by A, then B is an integer.
    B = 400 // A
    print(B)
else:
    # If 400 is not perfectly divisible by A, then no such integer B exists.
    print(-1)