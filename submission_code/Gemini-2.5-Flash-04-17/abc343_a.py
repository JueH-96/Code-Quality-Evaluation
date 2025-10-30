import sys

# Read the input line from standard input
line = sys.stdin.readline().split()

# Convert the split strings into integers A and B
A = int(line[0])
B = int(line[1])

# Calculate the sum of A and B
sum_ab = A + B

# The problem asks for any integer between 0 and 9 (inclusive)
# that is NOT equal to the sum A + B.
#
# The constraints specify that 0 <= A <= 9, 0 <= B <= 9, and A + B <= 9.
# This means the sum_ab is always an integer within the range [0, 9].
#
# We need to output an integer X such that 0 <= X <= 9 and X != sum_ab.
#
# A simple way to find such an integer is to consider the number 0.
# If sum_ab is not 0, then 0 is a valid answer because:
# 1. 0 is within the range [0, 9].
# 2. 0 is not equal to sum_ab (by definition of this case).
#
# If sum_ab *is* 0 (which only happens when A=0 and B=0, given the constraints),
# we cannot print 0. In this specific case, we need a different integer
# from the range [0, 9]. The number 1 is a valid choice because:
# 1. 1 is within the range [0, 9].
# 2. 1 is not equal to sum_ab (since sum_ab is 0).
#
# This logic covers all possible values of sum_ab (from 0 to 9) and guarantees
# an output that is within the required range [0, 9] and is not equal to the sum.

if sum_ab == 0:
    # If the sum is 0 (meaning A=0 and B=0), print any number from 1 to 9.
    # 1 is the simplest and smallest choice in the required range [0, 9].
    print(1)
else:
    # If the sum is not 0 (i.e., sum is 1, 2, ..., 9),
    # then 0 is a number in the required range [0, 9] that is not equal to the sum.
    print(0)