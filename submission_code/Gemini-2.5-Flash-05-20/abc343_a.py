# Read the two integers A and B from standard input
A, B = map(int, input().split())

# Calculate the sum of A and B
# According to the constraints, A + B will always be between 0 and 9, inclusive.
sum_ab = A + B

# We need to print any integer between 0 and 9 (inclusive) that is not equal to sum_ab.
# A simple strategy is to check if sum_ab is 0.
# If sum_ab is 0, then 1 is a valid number to print (since 1 != 0 and 0 <= 1 <= 9).
# If sum_ab is not 0 (meaning it's 1, 2, ..., or 9), then 0 is a valid number to print
# (since 0 != sum_ab and 0 <= 0 <= 9).

if sum_ab == 0:
    print(1)
else:
    print(0)