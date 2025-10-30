# YOUR CODE HERE
import sys

# Read the three integers A, B, C from standard input
line = sys.stdin.readline().split()
a = int(line[0])
b = int(line[1])
c = int(line[2])

# Store the numbers in a list
nums = [a, b, c]

# Sort the list in non-decreasing order
nums.sort()

# Assign sorted values to x, y, z such that x <= y <= z
x = nums[0]
y = nums[1]
z = nums[2]

# Check the conditions for partitioning into two or more groups with equal sums:

# Condition 1: Partition into two groups.
# This is possible if one element equals the sum of the other two.
# Since the list is sorted (x <= y <= z), the only possibility for this is if the largest element (z)
# equals the sum of the other two (x + y).
# The two groups would be {z} and {x, y}, both having a sum of z.
is_two_groups_possible = (z == x + y)

# Condition 2: Partition into three groups.
# This is possible only if all three elements are equal (x == y == z).
# The three groups would be {x}, {y}, {z}, each having a sum of x (or y or z).
is_three_groups_possible = (x == y and y == z)

# If either condition is met, it's possible to divide the numbers as required.
if is_two_groups_possible or is_three_groups_possible:
    print("Yes")
else:
    # If neither condition is met, it's not possible.
    print("No")